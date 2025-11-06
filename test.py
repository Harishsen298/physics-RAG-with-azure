# test_embed.py
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import AzureOpenAIEmbeddings
import os

emb = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION","2025-01-01-preview"),
    deployment=os.getenv("AZURE_EMBED_DEPLOYMENT")
)
vec = emb.embed_query("test vector")
print("OK, dims:", len(vec))
