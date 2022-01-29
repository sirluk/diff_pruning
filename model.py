import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize, parameters_to_vector
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel
)

from typing import Union, Callable, List, Dict, Generator, Tuple, Optional
from enum import Enum, auto

from training_logger import TrainLogger
from diff_param import DiffWeight, DiffWeightFixmask
from utils import dict_to_device


class ModelState(Enum):
    INIT = auto()
    FINETUNE = auto()
    FIXMASK = auto()


class DiffNetwork(torch.nn.Module):
    
    @property
    def device(self) -> torch.device:
        return next(self.pre_trained.parameters()).device
    
    @property
    def model_type(self) -> str:
        return self.pre_trained.config.model_type
    
    @property
    def model_name(self) -> str:
        return self.pre_trained.config._name_or_path
    
    @property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        for k in possible_keys:
            if k in self.pre_trained.config.__dict__:
                return getattr(self.pre_trained.config, k) + 2 # +2 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")
    
    @property
    def _parametrized(self) -> bool:
        return (self._model_state == ModelState.FINETUNE or self._model_state == ModelState.FIXMASK)
        
    @staticmethod
    # TODO log ratio could be removed if only symmetric concrete distributions are possible
    def get_log_ratio(concrete_lower: float, concrete_upper: float) -> int:
        # calculate regularization term in objective
        return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)
    
    @staticmethod
    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
        return torch.sigmoid(alpha - log_ratio).sum()

    
    def get_encoder_base_modules(self, return_names: bool = False):
        if self._parametrized:
            check_fn = lambda m: hasattr(m, "parametrizations")
        else:
            check_fn = lambda m: len(m._parameters)>0
        return [(n,m) if return_names else m for n,m in self.pre_trained.named_modules() if check_fn(m)]
    
    
    def get_layer_idx_from_module(self, n: str) -> int:
        # get layer index based on parameter name
        if self.model_type == "xlnet":
            search_str_emb = "word_embedding"
            search_str_hidden = "layer"
        else:
            search_str_emb = "embeddings"
            search_str_hidden = "encoder.layer"
        
        if search_str_emb in n:
            return 0
        elif search_str_hidden in n:
            return int(n.split(search_str_hidden + ".")[1].split(".")[0]) + 1
        else:
            return self.total_layers - 1
       
    
    def __init__(self, num_labels, *args, **kwargs): 
        super().__init__()
        self.num_labels = num_labels
        self.pre_trained = AutoModel.from_pretrained(*args, **kwargs)
        
        emb_dim = self.pre_trained.embeddings.word_embeddings.embedding_dim
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(.1),
            torch.nn.Linear(emb_dim, num_labels)
        )
        
        self._model_state = ModelState.INIT
            

    def forward(self, **x) -> torch.Tensor: 
        hidden = self.pre_trained(**x)[0][:,0]
        return self.classifier(hidden)

    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        alpha_init: Union[int, float],
        concrete_lower: float,
        concrete_upper: float,
        structured_diff_pruning: bool,
        gradient_accumulation_steps: int,
        num_epochs_finetune: int,
        num_epochs_fixmask: int,
        weight_decay: float,
        learning_rate: float,
        learning_rate_alpha: float,
        adam_epsilon: float,
        warmup_steps: int,
        sparsity_pen: Union[float,list],
        max_grad_norm: float,
        fixmask_pct: float,
        output_dir: Union[str, os.PathLike]
    ) -> None:
        self.global_step = 0
        
        num_epochs_total = num_epochs_finetune + num_epochs_fixmask
        train_steps_finetune = len(train_loader) // gradient_accumulation_steps * num_epochs_finetune
        train_steps_fixmask = len(train_loader) // gradient_accumulation_steps * num_epochs_fixmask
        log_ratio = self.get_log_ratio(concrete_lower, concrete_upper)
        
        self._add_diff_parametrizations(
            alpha_init,
            concrete_lower,
            concrete_upper,
            structured_diff_pruning  
        )
        
        self._init_optimizer_and_schedule(
            train_steps_finetune,
            learning_rate,
            weight_decay,
            adam_epsilon,
            warmup_steps,
            learning_rate_alpha,
        )
        
        self._init_sparsity_pen(sparsity_pen)
        
        self.zero_grad()
        
        train_str = "Epoch {}, model_state: {}{}"
        str_suffix = lambda result: ", " + ", ".join([f"validation {k}: {v}" for k,v in result.items()])
        result = {}
        
        train_iterator = trange(num_epochs_total, leave=False, position=0)
        for epoch in train_iterator:
            
            train_iterator.set_description(
                train_str.format(epoch, self._model_state, str_suffix(result)), refresh=True
            )
            
            # switch to fixed mask training
            if epoch == num_epochs_finetune:
                self._finetune_to_fixmask(fixmask_pct)
                self._init_optimizer_and_schedule(
                    train_steps_fixmask,
                    learning_rate,
                    weight_decay,
                    adam_epsilon
                )
            
            self._step(
                train_loader,
                loss_fn,
                logger,
                log_ratio,
                max_grad_norm,
                gradient_accumulation_steps
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )
            
            logger.validation_loss(epoch, result)
            
            # only save during fixmask training
            if self._model_state == ModelState.FIXMASK:
                if logger.is_best(result):
                    self._save_model_artefacts(
                        output_dir,
                        f"checkpoint-best-info.pt",
                        result["loss"],
                        epoch,
                        num_epochs_finetune
                    )
                    
            # count non zero
            n_p, n_p_zero = self._count_non_zero_params()            
            logger.non_zero_params(epoch, n_p, n_p_zero)
        
        print("Final results after " + train_str.format(epoch, self._model_state, str_suffix(result)))

                
    @torch.no_grad()   
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
    ) -> dict: 
        self.eval()

        output_list = []
        val_iterator = tqdm(val_loader, desc="evaluating", leave=False, position=1)
        for i, batch in enumerate(val_iterator):

            inputs, labels = batch
            inputs = dict_to_device(inputs, self.device)
            logits = self(**inputs)
            output_list.append((
                logits.cpu(),
                labels.cpu()
            ))
                    
        p, l = list(zip(*output_list))
        predictions = torch.cat(p, dim=0)
        labels = torch.cat(l, dim=0)
        
        eval_loss = loss_fn(predictions, labels).item()
        
        result = {metric_name: metric(predictions, labels) for metric_name, metric in metrics.items()}
        result["loss"] = eval_loss

        return result
            
            
    def _step(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        logger: TrainLogger,
        log_ratio: float,
        max_grad_norm: float,
        gradient_accumulation_steps: int
    ) -> None:
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}, loss without l0 pen: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels = batch
            inputs = dict_to_device(inputs, self.device)
            outputs = self(**inputs)
            loss = loss_fn(outputs, labels.to(self.device))
            
            loss_no_pen = loss.item()
            
            if self._model_state == ModelState.FINETUNE:
                l0_pen = 0.
                for module_name, base_module in self.get_encoder_base_modules(return_names=True):
                    layer_idx = self.get_layer_idx_from_module(module_name)
                    sparsity_pen = self.sparsity_pen[layer_idx]
                    module_pen = 0.
                    for n, par_list in list(base_module.parametrizations.items()):
                        for a in par_list[0].alpha_weights:
                            module_pen += self.get_l0_norm_term(a, log_ratio)
                    l0_pen += (module_pen * sparsity_pen)
                loss += l0_pen                   
                                      
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
                
            loss.backward()
            
            if ((step + 1) % gradient_accumulation_steps == 0) or ((step + 1) == len(epoch_iterator)):

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule
                
                self.zero_grad()
                
            logger.step_loss(self.global_step, loss, self.scheduler.get_last_lr()[0])
                           
            epoch_iterator.set_description(epoch_str.format(step, loss.item(), loss_no_pen), refresh=True)
            
            self.global_step += 1
                                    
                    
    def _save_model_artefacts(
        self,
        output_dir: Union[str, os.PathLike],
        filename: str,
        eval_loss: float,
        epoch: int,
        num_epochs_finetune: int
    ) -> None:
        output_dir = Path(output_dir)
        info_dict = {
            "epoch": epoch,
            "epochs_finetune": num_epochs_finetune,
            "epochs_fixmask": epoch - num_epochs_finetune + 1,
            "global_step": self.global_step,
            "model_state": self._model_state,
            "eval_loss": eval_loss,
            "encoder_state_dict": self.pre_trained.state_dict(),
            "clf_state_dict": self.classifier.state_dict()
        }
        torch.save(info_dict, Path(output_dir) / filename)
        
                
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        learning_rate: float,
        weight_decay: float,
        adam_epsilon: float,
        num_warmup_steps: int = 0,
        learning_rate_alpha: Optional[float] = None
    ) -> None:
        assert self._parametrized, "Optimizer can only be intialized with parametrized diff network."
        
        if self._model_state == ModelState.FINETUNE:
            optimizer_params = [
                {
                    # diff params (last dense layer is in no_decay list)
                    # TODO needs to be changed when original weight is set to fixed pre trained
                    "params": [p for n,p in self.pre_trained.named_parameters() if n[-8:]=="finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                "params": [p for n,p in self.pre_trained.named_parameters() if n[-5:]=="alpha" or n[-11:]=="alpha_group"],
                "lr": learning_rate_alpha
                },
                {
                    "params": self.classifier.parameters(),
                    "lr": learning_rate
                },
            ]
        elif self._model_state == ModelState.FIXMASK:
            optimizer_params = [{
                "params": self.parameters(),
                "lr": learning_rate
            }]            
        
        self.optimizer = AdamW(optimizer_params, eps=adam_epsilon)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        
        
    def _init_sparsity_pen(self, sparsity_pen: Union[float, List[float]]) -> None:        
        if isinstance(sparsity_pen, list):
            self.sparsity_pen = sparsity_pen
            assert len(sparsity_pen) == self.total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
        else:
            self.sparsity_pen = [sparsity_pen] * self.total_layers
            
                
    def _add_diff_parametrizations(self, *args) -> None:
        assert not self._parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():
            for n,p in list(base_module.named_parameters()):
                parametrize.register_parametrization(base_module, n, DiffWeight(p, *args))   
        self._model_state = ModelState.FINETUNE
        
      
    @torch.no_grad()
    def _finetune_to_fixmask(self, pct: float) -> None:
        diff_values = torch.tensor([])
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                par_list[0].set_mode(train=False)
                diff_val = (getattr(base_module, n) - par_list.original).detach().cpu()
                diff_values = torch.cat([diff_values, diff_val.flatten()])
                
        k = int(len(diff_values) * pct)
        cutoff = torch.topk(torch.abs(diff_values), k, largest=True, sorted=True)[0][-1]
                
        for base_module in self.get_encoder_base_modules():
            for n, par_list in list(base_module.parametrizations.items()):
                pre_trained = torch.clone(par_list.original)
                parametrize.remove_parametrizations(base_module, n)
                p = base_module._parameters[n].detach()
                diff_weight = (p - pre_trained)
                diff_mask = (torch.abs(diff_weight) >= cutoff)
                base_module._parameters[n] = Parameter(diff_weight * diff_mask)
                parametrize.register_parametrization(base_module, n, DiffWeightFixmask(pre_trained, diff_mask))
        self._model_state = ModelState.FIXMASK
                                        
    
    @torch.no_grad() 
    def _count_non_zero_params(self) -> Tuple[int, int]:
        n_p = 0
        n_p_zero = 0
        base_modules = self.get_encoder_base_modules()
        if self._parametrized:
            for base_module in base_modules:
                for n, par_list in list(base_module.parametrizations.items()):
                    if isinstance(par_list[0], DiffWeightFixmask):
                        n_p += par_list[0].mask.numel()
                        n_p_zero += ~par_list[0].mask.sum()
                    else:
                        par_list[0].set_mode(train=False)
                        z = par_list[0].z.detach()
                        n_p += z.numel()
                        n_p_zero += torch.isclose(z.float().cpu(), torch.tensor([0.]), atol=1e-10).sum()
                        par_list[0].set_mode(train=True)
        else:
            for p in self.pre_trained.parameters():
                n_p += p.numel()
                n_p_zero += torch.isclose(p.float().cpu(), torch.tensor([0.])).sum()                                    
        return n_p, n_p_zero

    
    def _remove_diff_parametrizations(self) -> None:
        for parametrized_module in self.get_encoder_base_modules():
            for n in list(parametrized_module.parametrizations):
                parametrize.remove_parametrizations(parametrized_module, n)
        self._model_state = ModelState.INIT
