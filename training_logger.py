from pathlib import Path
from typing import Union, Optional
import logging
import os
import sys
import math

from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)


class TrainLogger:
    delta: float = 1e-8
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        logging_step: int,
        logger_name: Optional[str] = None
    ):
        assert logging_step > 0, "logging_step needs to be > 0"
        
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
            
        # setup logger
        # logger_name = f"{__name__}.{logger_name}" if logger_name else __name__
        # self.logger = logging.getLogger(logger_name)
        # # add file handler
        # fh = logging.FileHandler(log_dir / f'{logger_name}.log')
        # fh.setLevel(logging.INFO)
        # self.logger.addHandler(fh)
        # add stdout
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setLevel(logging.INFO)
        # self.logger.addHandler(sh)
        
        self.logging_step = logging_step

        self.writer = SummaryWriter(log_dir / 'tensorboard' / logger_name)

        self.steps = 0
        self.logging_loss = 0.
        self.best_eval_loss = math.inf

    def validation_loss(self, eval_step: int, result: dict): 
        # self.logger.info(f"***** Eval results for step {self.eval_steps} *****")
        metrics_str = []
        for name, value in sorted(result.items(), key=lambda x: x[0]):
            self.writer.add_scalar(f'val/{name}', value, eval_step)
            # self.logger.info(f"{name} = {value}")

            
    def is_best(self, result: dict):
        if result["loss"] < self.best_eval_loss + self.delta:
            self.best_eval_loss = result["loss"]
            return True
        
            
    def step_loss(self, step: int, loss: float, lr: float):
        
        self.logging_loss += loss
        self.steps += 1
        
        if step % self.logging_step == 0:
 
            logs = {
                "step_learning_rate": lr,
                "step_loss": self.logging_loss / self.steps
            }
            for key, value in logs.items():
                self.writer.add_scalar(f'train/{key}', value, step)
                
            self.logging_loss = 0.
            self.steps = 0
            
    def non_zero_params(self, step, n_params, n_params_zero):
        zero_ratio = n_params_zero / n_params
        self.writer.add_scalar("train/zero_ratio", zero_ratio, step)
