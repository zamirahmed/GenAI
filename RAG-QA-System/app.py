import streamlit as st
from qa import ask_question


st.set_page_config(
    page_title="iConsultera AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("iConsultera AI Assistant")


question = st.chat_input("Ask something about your document")

if question:

    st.chat_message("user").write(question)

    with st.chat_message("assistant"):

        with st.spinner("loading"):

            response = ask_question(question)

        st.write(str(response))