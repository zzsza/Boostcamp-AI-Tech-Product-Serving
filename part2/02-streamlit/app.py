import streamlit as st
import pandas as pd
from predict import load_model, get_prediction

def main():
    st.title('Semantic Textual Similarity')
    st.caption('*원하는 문장 두 개를 입력해보세요!*:sunglasses:')

    model = load_model()
    model.eval()

    # 측정 결과들 모아두는 df
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame({
            'sentence 1': [],
            'sentence 2' : [],
            'similarity' : []
        })

    with st.form(key='문장입력 form'):
        sentence1 = st.text_input("Enter Sentence 1:")
        sentence2 = st.text_input("Enter Sentence 2:")
        form_submitted = st.form_submit_button('유사도 측정')

    if form_submitted:
        if sentence1 and sentence2:
            similarity_score = get_prediction(model, sentence1, sentence2)

            # df에 이미 있는 유사도 쌍이면 추가 안함
            if not ((st.session_state.df['sentence 1'] == sentence1) & (st.session_state.df['sentence 2'] == sentence2)).any():
                # 새로운 데이터를 기존 df에 합치기 
                new_data = pd.DataFrame({
                    'sentence 1': [sentence1],
                    'sentence 2': [sentence2],
                    'similarity': [similarity_score]
                })
                st.session_state.df = st.session_state.df.append(new_data, ignore_index=True)
                
                # similarity 기준으로 순위 매기기
                st.session_state.df = st.session_state.df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
                
                # rank 컬럼 추가
                st.session_state.df['rank'] = st.session_state.df.index + 1

            st.write(f"두 문장의 유사도 : {similarity_score}")
            st.success('성공!')
        else:
            st.write("Please enter both sentences.")
            st.error('다시 한번 생각해보세요!')

    st.divider()
    col1, col2, col3 = st.columns(3)
    
    # df 크기 조절
    col1.checkbox("창 크기조절", value=True, key="use_container_width")

    # df 리셋 버튼
    if col2.button("데이터 리셋하기"):
        st.session_state.df = pd.DataFrame({
            'sentence 1': [],
            'sentence 2' : [],
            'similarity' : []
        })

    # df csv로 다운로드
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False, header=True).encode('cp949')
    csv = convert_df(st.session_state.df)
    col3.download_button(
        label="CSV로 다운받기",
        data=csv,
        file_name='sts_data_outputs.csv',
        mime='text/csv',
    )

    st.dataframe(st.session_state.df, use_container_width=st.session_state.use_container_width)

main()
