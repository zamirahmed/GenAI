import os

from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)

from llama_index.embeddings.google_genai import (
    GoogleGenAIEmbedding,
)

from llama_index.llms.google_genai import (
    GoogleGenAI,
)


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing.")


# ============================================================
# 2. Embedding model
# ============================================================

embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=api_key,
)


# ============================================================
# 3. Gemini LLM
# ============================================================

llm = GoogleGenAI(
    model="gemini-3.1-flash-lite",
    api_key=api_key,
)


# ============================================================
# 4. Load persisted index
# ============================================================

storage_context = StorageContext.from_defaults(
    persist_dir="index"
)

index = load_index_from_storage(
    storage_context,
    embed_model=embed_model,
)


# ============================================================
# 5. Query engine
# ============================================================

query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
)


# ============================================================
# 6. Ask question
# ============================================================

def ask_question(question):

    response = query_engine.query(question)

    return response