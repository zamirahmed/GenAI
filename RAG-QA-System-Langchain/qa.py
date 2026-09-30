import os

from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Configuration
# ============================================================

INDEX_DIR = "index"

COLLECTION_NAME = "qa_collection"

EMBEDDING_MODEL = "gemini-embedding-001"

LLM_MODEL = "gemini-3.8-flash"

TOP_K = 4


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing from .env"
    )


# ============================================================
# Create embedding model
# ============================================================

embed_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
)


# ============================================================
# Load Chroma vector database
# ============================================================

vectorstore = Chroma(
    persist_directory=INDEX_DIR,
    embedding_function=embed_model,
    collection_name=COLLECTION_NAME,
)


# ============================================================
# Create retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": TOP_K
    }
)


# ============================================================
# Create Gemini LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,
)


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a professional question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

Rules:

1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not available in the context,
   say:

   "I don't have enough information in the knowledge
   base to answer this question."

4. Give a clear and useful answer.
5. Use bullet points when appropriate.

Context:
-------------------------
{context}
-------------------------

Question:
{question}

Answer:
"""
)


# ============================================================
# Format retrieved documents
# ============================================================

def format_documents(documents):

    context_parts = []

    for document in documents:

        filename = document.metadata.get(
            "file_name",
            "Unknown",
        )

        file_path = document.metadata.get(
            "file_path",
            "",
        )

        source = filename

        if file_path:
            source = file_path

        context_parts.append(
            f"Source: {source}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# Ask question
# ============================================================

def ask_question(question):

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    documents = retriever.invoke(
        question
    )

    if not documents:

        return {
            "answer": (
                "I don't have enough information "
                "in the knowledge base to answer "
                "this question."
            ),
            "sources": [],
        }


    # --------------------------------------------------------
    # Create context
    # --------------------------------------------------------

    context = format_documents(
        documents
    )


    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )


    # --------------------------------------------------------
    # Ask Gemini
    # --------------------------------------------------------

    response = llm.invoke(
        messages
    )

    answer = response.content

    # Make sure Streamlit receives plain text
    if isinstance(answer, str):

        answer = answer.strip()

    elif isinstance(answer, list):

        text_parts = []

        for item in answer:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        answer = "\n".join(
            text_parts
        ).strip()

    else:

        answer = str(answer).strip()


    # --------------------------------------------------------
    # Collect sources
    # --------------------------------------------------------

    sources = []

    for document in documents:

        file_path = document.metadata.get(
            "file_path",
            document.metadata.get(
                "file_name",
                "Unknown",
            ),
        )

        if file_path not in sources:

            sources.append(
                file_path
            )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
    }