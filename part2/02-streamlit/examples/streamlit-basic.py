import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("Title")
st.header("Header")
st.subheader("subheader")


st.write("Write Something")

if st.button("버튼이 클릭되면"):
    st.write("클릭 후 보이는 메세지!")

if st.button("버튼이 클릭되면2"):
    st.write("클릭 후 보이는 메세지2!")

checkbox_btn = st.checkbox('체크박스 버튼')

if checkbox_btn:
    st.write('체크박스 버튼 클릭!')

checkbox_btn2 = st.checkbox('Checkbox Button2', value=True)

if checkbox_btn2:
    st.write('Button2')


df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

st.markdown("========")


st.write(df)
st.dataframe(df)
st.table(df)

st.dataframe(df.style.highlight_max(axis=0))
st.table(df.style.highlight_max(axis=0))

st.metric("My metric", 42, 2)

st.json(df.to_json())


chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c']
)

st.line_chart(chart_data)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


selected_item = st.radio("Radio Part", ("A", "B", "C"))

if selected_item == "A":
    st.write("A!!")
elif selected_item == "B":
    st.write("B!")
elif selected_item == "C":
    st.write("C!")

option = st.selectbox('Please select in selectbox!',
                      ('kyle', 'seongyun', 'zzsza'))

st.write('You selected:', option)

multi_select = st.multiselect('Please select somethings in multi selectbox!',
                              ['A', 'B', 'C', 'D'])

st.write('You selected:', multi_select)

values = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

text_input = st.text_input("텍스트를 입력해주세요")
st.write(text_input)


password_input = st.text_input("암호를 입력해주세요", type="password")


number_input = st.number_input("숫자를 입력해주세요")
st.write(number_input)

st.date_input("날짜를 입력하세요")
st.time_input("시간을 입력하세요")

st.caption("This is caption")
st.code("a = 123")
st.latex("\int a x^2 \,dx")


st.sidebar.button("hi")


col1, col2, col3, col4 = st.columns(4)
col1.write("this is col1")
col2.write("this is col2")
col3.write("this is col3!!!")
col4.write("this is col4~~~")


with st.expander("클릭하면 열려요"):
    st.write("content!")


with st.spinner("Please wait.."):
    time.sleep(5)


st.balloons()

st.success("Success")
st.info("Info")
st.warning("Warning")
st.error("Error message")


with st.form(key="입력 form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    st.form_submit_button("login")


uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

# session_state가 없다면
st.title('Counter Example without session state')

count_value = 0

increment = st.button('Increment')
if increment:
    count_value += 1

decrement = st.button('Decrement')
if decrement:
    count_value -= 1

st.write('Count = ', count_value)


# session_state가 있다면
st.title('Counter Example with session state')

# count session_state에 init
if 'count' not in st.session_state:
    st.session_state.count = 0

# increment 버튼이 클릭되면 session_state의 count에 1을 더함
increment = st.button('Increment1')
if increment:
    st.session_state.count += 1

# decrement 버튼이 클릭되면 session_state의 count에 1을 더함
decrement = st.button('Decrement2')
if decrement:
    st.session_state.count -= 1

st.write('Count = ', st.session_state.count)


DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data



