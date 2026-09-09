# app.py
import os
import yaml
from dotenv import load_dotenv

from utils.loader import load_and_chunk_docs
from utils.retriever import (
    get_retriever,
    load_config,
)
from utils.query_rewriter import rewrite_query
from utils.hyde_generator import generate_hyde_embedding

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# ======================================================
# Load config + Get correct LLM (OpenAI | Gemini)
# ======================================================
def get_llm(config):
    provider = config["llm"]["provider"]

    if provider == "openai":
        return ChatOpenAI(
            model=config["llm"]["model_openai"],
            temperature=config["llm"]["temperature"]
        )

    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config["llm"]["model_gemini"],
            temperature=config["llm"]["temperature"]
        )

    else:
        raise ValueError("Provider must be one of: openai | gemini")


# ======================================================
# RAG Pipelines
# ======================================================

# --- Baseline retrieval ---
def run_baseline(query, retriever, llm):
    docs = retriever.invoke(query) #getting the relevant documents from the retriever based on the user query

    context = "\n".join([d.page_content for d in docs])

    prompt = [
        ("system", "Use ONLY the provided context to answer the question."),
        ("human", f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
    ]

    response = llm.invoke(prompt).content.strip()
    return response, docs


# --- Query Rewriting retrieval ---
def run_rewriting(query, retriever, llm):
    rewritten = rewrite_query(query)
    print(f"\n🔁 Rewritten Query → {rewritten}")

    #getting the relevant documents from the retriever (vector db) based on the rewritten query
    docs = retriever.invoke(rewritten) 
    
    context = "\n".join([d.page_content for d in docs])

    # The prompt instructs the LLM to answer the question using only the provided context from the 
    # retrieved documents and rewritten query. 
    # The LLM is expected to generate a response based on the context and the rewritten query.
    prompt = [
        ("system", "Use only the retrieved context to answer the question."),
        ("human", f"Context:\n{context}\n\nQuestion: {rewritten}\nAnswer:")
    ]

    response = llm.invoke(prompt).content.strip()
    return response, docs


# --- HyDE retrieval ---
def run_hyde(query, retriever, llm):
    hyde_emb = generate_hyde_embedding(query)

    # HyDE uses vector similarity search directly to fetch relevant documents based on the embedding 
    # of the hypothetical answer generated from the user query.
    docs = retriever.vectorstore.similarity_search_by_vector(hyde_emb, k=5)
    context = "\n".join([d.page_content for d in docs])

    prompt = [
        ("system", "Answer using ONLY the provided context."),
        ("human", f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
    ]

    response = llm.invoke(prompt).content.strip()
    return response, docs


# ======================================================
# MAIN PROGRAM
# ======================================================
def main():
    config = load_config()
    llm = get_llm(config)

    # Build or load FAISS based on provider
    provider = config["llm"]["provider"]
    print(f"\n🚀 Active Provider: {provider.upper()}")

    # If FAISS index doesn't exist → build from chunks
    chunks = load_and_chunk_docs("./data/raw/insurance_docs")
    retriever = get_retriever(config, chunks_if_needed=chunks)

    # User Query
    query = input("\n🔍 Enter your question: ").strip()

    # --------------------------
    # BASELINE
    # --------------------------
    print("\n====================")
    print("✅ BASELINE RAG")
    print("====================")

    baseline_ans, _ = run_baseline(query, retriever, llm)
    print(f"Query: {query}")
    print(f"Ans - {baseline_ans}")

    # --------------------------
    # QUERY REWRITING RAG
    # --------------------------
    print("\n==============================")
    print("✅ QUERY REWRITING + RAG")
    print("==============================")

    rewriting_ans, _ = run_rewriting(query, retriever, llm)
    print(f"Ans - {rewriting_ans}")

    # --------------------------
    # HYDE RAG
    # --------------------------
    print("\n====================")
    print("✅ HYDE RAG")
    print("====================")

    hyde_ans, _ = run_hyde(query, retriever, llm)
    print(f"\n🤖hyde+retriever Ans - {hyde_ans}")


if __name__ == "__main__":
    main()