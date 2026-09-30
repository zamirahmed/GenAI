import os

import streamlit as st

from qa import ask_question


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="RAG Question Answering",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# Header
# ============================================================

st.title(
    "🤖 RAG Question Answering System"
)

st.caption(
    "LangChain + Gemini + ChromaDB"
)


# ============================================================
# Check ChromaDB
# ============================================================

if not os.path.exists("index"):

    st.warning(
        "Knowledge base does not exist yet."
    )

    st.info(
        "Please run the following command first:"
    )

    st.code(
        "python ingest.py"
    )

    st.stop()


# ============================================================
# Question Input
# ============================================================

question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What services does the company provide?"
    ),
)


# ============================================================
# Ask Question
# ============================================================

if st.button(
    "Ask Question",
    type="primary",
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching knowledge base..."
        ):

            try:

                result = ask_question(
                    question
                )

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )

                st.stop()


        # ====================================================
        # Answer
        # ====================================================

        st.subheader(
            "Answer"
        )

        st.write(
            result["answer"]
        )


        # ====================================================
        # Sources
        # ====================================================

        if result["sources"]:

            st.subheader(
                "Sources"
            )

            for source in result["sources"]:

                st.write(
                    f"📄 {source}"
                )