# utils/query_rewriter.py
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_llm_for_rewriting(config):
    provider = config["llm"]["provider"]
    temperature = config["llm"]["temperature"]

    if provider == "openai":
        return ChatOpenAI(
            model=config["llm"]["model_openai"],
            temperature=temperature
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config["llm"]["model_gemini"],
            temperature=temperature
        )
    else:
        raise ValueError("Unsupported provider. Choose 'openai' or 'gemini'.")
    
def rewrite_query(user_query: str, config_path="config.yaml"):
    """
    Rewrites or expands a query to be more specific, contextual, and
    retrieval-friendly. Works with OpenAI or Gemini.
    """
    config = load_config(config_path)
    llm = get_llm_for_rewriting(config)

    system_prompt = (
        "You are an expert at improving user queries for retrieval-based search. "
        "Rewrite the query to make it clearer, longer, and more specific, "
        "while keeping the original intent."
    )
    
    prompt = [
        ("system", system_prompt),
        ("human", f"Rewrite the following query:\n\n{user_query}")
    ]
    
    rewritten_query = llm.invoke(prompt).content.strip()
    return rewritten_query