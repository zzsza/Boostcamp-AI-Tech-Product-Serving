import torch

import pytorch_lightning as pl
import transformers

from config import config
from model import Model


def load_model(model_path: str = config.model_path):
    return torch.load(model_path)


def get_trainer(max_epoch: int = 10):
    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch, log_every_n_steps=1)
    else:
        trainer = pl.Trainer(accelerator="cpu", max_epochs=max_epoch, log_every_n_steps=1)
    return trainer


def get_tokenizer(model_name: str = 'klue/roberta-small'):
    return transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)
