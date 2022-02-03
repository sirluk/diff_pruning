import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup

from typing import Union, Callable, Dict

from src.utils import dict_to_device
from src.training_logger import TrainLogger


class BaselineNetwork(torch.nn.Module):
    
    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device
    
    @property
    def model_type(self) -> str:
        return self.encoder.config.model_type
    
    @property
    def model_name(self) -> str:
        return self.encoder.config._name_or_path
    
    @property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        for k in possible_keys:
            if k in self.encoder.config.__dict__:
                return getattr(self.encoder.config, k) + 2 # +2 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")

    
    def __init__(self, num_labels, *args, **kwargs): 
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(*args, **kwargs)
        
        emb_dim = self.encoder.embeddings.word_embeddings.embedding_dim
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(.1),
            torch.nn.Linear(emb_dim, num_labels)
        )
            

    def forward(self, **x) -> torch.Tensor:
        hidden = self.encoder(**x)[0][:,0]
        return self.classifier(hidden)

    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainLogger,
        loss_fn: Callable,
        metrics: Dict[str, Callable],
        num_epochs: int,
        weight_decay: float,
        learning_rate: float,
        adam_epsilon: float,
        warmup_steps: int,
        max_grad_norm: float
    ) -> None:
        self.global_step = 0

        train_steps = len(train_loader) * num_epochs
        
        self._init_optimizer_and_schedule(
            train_steps,
            weight_decay,
            learning_rate,
            adam_epsilon,
            warmup_steps
        )
        
        self.zero_grad()
        
        train_str = "Epoch {}{}"
        str_suffix = lambda result: ", " + ", ".join([f"validation {k}: {v}" for k,v in result.items()])
        result = {}
        
        train_iterator = trange(num_epochs, leave=False, position=0)
        for epoch in train_iterator:
            
            train_iterator.set_description(
                train_str.format(epoch, str_suffix(result)), refresh=True
            )
            
            self._step(
                train_loader,
                loss_fn,
                logger,
                max_grad_norm
            )
                        
            result = self.evaluate(
                val_loader, 
                loss_fn,
                metrics
            )
            
            logger.validation_loss(epoch, result)
        
        print("Final results after " + train_str.format(epoch, str_suffix(result)))

                
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
        max_grad_norm: float
    ) -> None:
        self.train()
        
        epoch_str = "training - step {}, loss: {:7.5f}"
        epoch_iterator = tqdm(train_loader, desc=epoch_str.format(0, math.nan), leave=False, position=1)
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels = batch
            inputs = dict_to_device(inputs, self.device)
            outputs = self(**inputs)
            loss = loss_fn(outputs, labels.to(self.device))
                
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
                
            self.zero_grad()
                
            logger.step_loss(self.global_step, loss, self.scheduler.get_last_lr()[0])
                           
            epoch_iterator.set_description(epoch_str.format(step, loss.item()), refresh=True)
            
            self.global_step += 1

                
    def _init_optimizer_and_schedule(
        self,
        num_training_steps: int,
        weight_decay: float,
        learning_rate: float,
        adam_epsilon: float,
        num_warmup_steps: int = 0
    ) -> None:
        optimizer_params = [
            {
                "params": self.encoder.parameters(),
                "weight_decay": weight_decay,
                "lr": learning_rate
            },
            {
                "params": self.classifier.parameters(),
                "lr": learning_rate
            }
        ]
        
        self.optimizer = AdamW(optimizer_params, eps=adam_epsilon)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
