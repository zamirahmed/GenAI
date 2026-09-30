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

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


embed_model = GoogleGenAIEmbedding(
    model_name ='gemini-embedding-001',
    api_key=api_key,
)

llm = GoogleGenAI(
    model="gemini-3.1-flash-lite",
)
storage_context = StorageContext.from_defaults(
    persist_dir="index"
)

index = load_index_from_storage(
        storage_context,
        embed_model=embed_model
    )

# ----------------------------------------
# Query engine
# ----------------------------------------

query_engine =  index.as_query_engine(
    llm=llm,
    similarity_top_k=4
)

def ask_question(question):

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)
    print("=" * 70)

    retriever = index.as_retriever(
        similarity_top_k=4
    )
    retrieved_nodes = retriever.retrieve(question)
    print("\nRETRIEVED DOCUMENTS:")
    print("-" * 70)

    for i, node in enumerate(retrieved_nodes, start=1):

        print(f"\n--- Result {i} ---")
        print("Score:", node.score)
        print("Text:")
        print(node.text)

    print("\n" + "=" * 70)


    response = query_engine.query(question)

    print("\nFINAL ANSWER:")
    print(response)

    print("=" * 70)
    
    return response
