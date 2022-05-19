import torch
import streamlit as st
from typing import *
from transformers import GPT2Config, GPT2LMHeadModel
# from mymodel import MyChatbot, MyTokenizer
from tokenizers import SentencePieceBPETokenizer

torch.manual_seed(42)

# @st.cache
def load_model() -> GPT2LMHeadModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPT2Config(vocab_size=50000)
    model = GPT2LMHeadModel(config).to(device)
    model.load_state_dict(torch.load("chitchat_model.pth", map_location=device))
    return model

def load_tokenizer() -> SentencePieceBPETokenizer:
    tokenizer = SentencePieceBPETokenizer("./tokenizerUtils/vocab.json", "./tokenizerUtils/merges.txt")
    tokenizer.add_special_tokens(['<s>', '</s>'])
    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=128)
    
    return tokenizer

def get_prediction(input_sent:str, model:GPT2LMHeadModel, tokenizer:SentencePieceBPETokenizer) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def encoding(text):
        text = '<s>'+text+'</s><s>'
        return torch.tensor(tokenizer.encode(text).ids).unsqueeze(0).to(device)

    def decoding(ids):
        return tokenizer.decode_batch(ids)

    tokenizer.no_padding()
    tokenizer.no_truncation()

    e_s = tokenizer.token_to_id('</s>')
    unk = tokenizer.token_to_id('<unk>')
    
    input_ids = encoding(input_sent)

    sample_outputs = model.generate(
        input_ids,
        num_return_sequences=5,
        do_sample=True, 
        max_length=128, 
        top_k=50, 
        top_p=0.95, 
        eos_token_id=e_s,
        early_stopping=True,
        bad_words_ids=[[unk]]  # 생성 단어로 unk토큰이 선택될 경우 다른 단어를 선택하도록 설정
    )

    decoded_result = decoding(sample_outputs.tolist())
    return decoded_result[0]