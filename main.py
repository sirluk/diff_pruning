# import IPython; IPython.embed(); exit(1)

import ruamel.yaml as yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from transformers import AutoTokenizer
from datasets import load_dataset

from src.model import DiffNetwork
from src.training_logger import TrainLogger
from src.metrics import accuracy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_ds(tokenizer) -> TensorDataset:
    
    ds = load_dataset("glue", "sst2", cache_dir="cache")
    ds = ds.map(
        lambda x: tokenizer(x["sentence"], padding="max_length", max_length=128, truncation=True),
        batched=True,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset"
    )
    

def get_ds_part(ds, part) -> TensorDataset:
    _ds = ds[part]
    return TensorDataset(
        torch.tensor(_ds["input_ids"], dtype=torch.long),
        torch.tensor(_ds["attention_mask"], dtype=torch.long),
        torch.tensor(_ds["token_type_ids"], dtype=torch.long),
        torch.tensor(_ds["label"], dtype=torch.float)
    )


def main(args):    
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    ds = get_ds(tokenizer)
    
    pred_fn = lambda x: (torch.sigmoid(x) > .5).long()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = {
        "acc": lambda x, y: accuracy(pred_fn(x), y),
        "balanced_acc": lambda x, y: accuracy(pred_fn(x), y, balanced=True)
    }
   
    ds_train = get_ds_part(ds, "train")
    train_loader = DataLoader(ds_train, sampler=RandomSampler(ds_train), batch_size=args.batch_size)
    ds_eval = get_ds_part(ds, "validation")
    eval_loader = DataLoader(ds_eval, sampler=SequentialSampler(ds_eval), batch_size=args.batch_size)

    train_logger = TrainLogger(
        log_dir = Path(args.output_dir) / "logs",
        logger_name = "diff_pruning_logger",
        logging_step = args.logging_step
    )

    trainer = DiffNetwork(1, args.model_name)
    trainer.to(DEVICE)

    trainer.fit(
        train_loader,
        eval_loader,
        train_logger,
        loss_fn,
        metrics,
        args.alpha_init,
        args.concrete_lower,
        args.concrete_upper,
        args.structured_diff_pruning,
        args.gradient_accumulation_steps,
        args.num_epochs_finetune,
        args.num_epochs_fixmask,
        args.weight_decay,
        args.learning_rate,
        args.learning_rate_alpha,
        args.adam_epsilon,
        args.warmup_steps,
        args.sparsity_pen,
        args.max_grad_norm,
        args.fixmask_pct,
        args.output_dir
    )


if __name__ == "__main__":
        
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)    
    args = argparse.Namespace(**cfg["train_config"])

    main(args)


