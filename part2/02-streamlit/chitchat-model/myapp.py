import streamlit as st
import torch
from mypredict import load_model, load_tokenizer, get_prediction
torch.manual_seed(42)

# SETTING PAGE CONFIG TO WIDE MODE
# st.set_page_config(layout="wide")

def main():
    st.title("Streamlit ChatBot")

    model = load_model()
    tokenizer = load_tokenizer()
    model.eval()

    uploaded_text = st.text_input("대화를 시작하세요.")
    if uploaded_text:
        input_sent = uploaded_text

        st.write("Understanding chat...")

        answer = get_prediction(input_sent, model=model, tokenizer=tokenizer)
        answered_text = st.text_input("챗봇의 대답:", answer)

main()