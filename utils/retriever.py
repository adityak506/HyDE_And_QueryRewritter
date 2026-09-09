# utils/retriever.py
import os
import yaml
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def get_embedding_model(provider, config):
    if provider == "openai":
        return OpenAIEmbeddings(model=config["embedding"]["openai_model"])
    elif provider == "gemini":
        return GoogleGenerativeAIEmbeddings(model=config["embedding"]["gemini_model"])
    else:
        raise ValueError("Provider must be one of: openai | gemini")
    
def get_index_path(provider, config):
    if provider == "openai":
        return config["vectordb"]["faiss_openai"]
    else:
        return config["vectordb"]["faiss_gemini"]
#index_path is the path where the FAISS index is stored. From where retriever retrieves the vector 
# embeddings of the documents. It is specified in the config.yaml file under vectordb section.

#we use this function, when we need to create retriever when we need to build the vector db first and then
#create the retriever and return that.
def create_retriever(chunks, provider, config):
    index_path = get_index_path(provider, config)
    embedding_model = get_embedding_model(provider, config)

    os.makedirs(index_path, exist_ok=True)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(index_path)

    print(f"✅ Created new FAISS index ({provider}) at: {index_path}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval"]["top_k"]})
    return retriever
#create_retriever function creates a retriever object that can be used to retrieve relevant documents
# based on a query. It takes the document chunks, the provider (either "openai" or "gemini"), and
# the configuration as input. It creates a FAISS index from the document chunks and saves it locally.
# The retriever is then created from the vectorstore and returned.

#load_retriever function loads an existing FAISS index from the specified path and creates a retriever object.
def load_retriever(provider, config):
    index_path = get_index_path(provider, config)
    embedding_model = get_embedding_model(provider, config)

    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print(f"✅ Loaded FAISS index ({provider}) from: {index_path}")
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval"]["top_k"]})
    return retriever

#chunks_if_needed becuase what if my vector db already exist then we will not need it
def get_retriever(config, chunks_if_needed=None):
    provider = config["llm"]["provider"]
    index_path = get_index_path(provider, config)

#if index does not exist, we will create a new one using the provided chunks. If the index already 
# exists, we will load the existing retriever.
    if not os.path.exists(index_path):
        print(f"⚠️ No FAISS index found for provider '{provider}'. Creating one...\n")

        if chunks_if_needed is None:
            raise RuntimeError(
                "Chunks not provided. The caller must supply chunks when index doesn't exist."
            )

        return create_retriever(chunks_if_needed, provider, config)

    return load_retriever(provider, config)