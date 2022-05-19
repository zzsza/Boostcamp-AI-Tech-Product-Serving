import torch
from tokenizers import SentencePieceBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel
import pandas as pd
from transformers import AdamW
from torch.utils.data import DataLoader

class MyChatbot(GPT2LMHeadModel):
    def __init__(self, config) :
        super(GPT2LMHeadModel, self).__init__(config)

class MyTokenizer(SentencePieceBPETokenizer):
    def __init__(self) :
        super(SentencePieceBPETokenizer, self).__init__()