# app.py
import os
import logging
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

# ---- LangChain split packages (no langchain.chains used) ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_app")

# ─── Helpers ─────────────────────────────────────────────────
def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    logger.info(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(pages)
    logger.info(f"Split into {len(docs)} chunks")
    return docs

def build_faiss_index(docs, embed_model, index_path="faiss_index"):
    # rebuild each start to keep it simple
    if os.path.isdir(index_path):
        import shutil
        shutil.rmtree(index_path)
        logger.info("Removed existing FAISS index")
    vs = FAISS.from_documents(docs, embed_model)
    vs.save_local(index_path)
    logger.info("FAISS index built and saved")
    return vs

def make_retriever(vs, k=2):
    logger.info(f"Creating retriever with k={k}")
    return vs.as_retriever(search_kwargs={"k": k})

def format_context(docs):
    # keep it compact; you can include sources separately
    return "\n\n".join(d.page_content for d in docs)

# ─── App init ────────────────────────────────────────────────
load_dotenv()

AOAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")        # e.g. https://<res>.openai.azure.com/
AOAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
AOAI_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
EMB_DEPLOYMENT  = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")

PDF_PATH   = os.getenv("PDF_FILE_PATH", "physics_semiconductor.pdf")
TEMP       = float(os.getenv("CHAT_TEMPERATURE", 0.3))

if not (AOAI_ENDPOINT and AOAI_API_KEY):
    raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env")

logger.info("Initializing Azure OpenAI clients")
llm = AzureChatOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    api_version=AOAI_API_VER,
    deployment_name=CHAT_DEPLOYMENT,
    temperature=TEMP,
)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    api_version=AOAI_API_VER,
    deployment=EMB_DEPLOYMENT,
)

# Build index & retriever ONCE
docs      = load_and_split_pdf(PDF_PATH)
vs        = build_faiss_index(docs, embeddings)
retriever = make_retriever(vs, k=2)

# ─── Flask ──────────────────────────────────────────────────
app = Flask(__name__)  # no static_folder

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data  = request.get_json(force=True)
        query = (data.get("query") or "").strip()
        logger.info(f"Query received: {query!r}")

        # 1) Greeting / empty → direct LLM
        if query.lower() in {"", "hi", "hello", "hey"}:
            resp = llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=query or "Hello!")
            ])
            return jsonify(answer=resp.content, sources=[])

        # 2) Retrieve relevant docs
        hits = hits = retriever.invoke(query)          # returns List[Document]
        logger.info(f"RAG hits: {len(hits)}")

        if not hits:
            # Fallback to direct LLM
            resp = llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=query)
            ])
            return jsonify(answer=resp.content, sources=[])

        # 3) Ask the model with context
        context = format_context(hits)
        system = (
            "You are a helpful physics tutor. "
            "Use ONLY the provided context to answer. "
            "If the answer cannot be found in the context, say you don't know."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        answer = resp.content

        sources = [
            f"page {d.metadata.get('page','?')}: {d.page_content[:100].replace(chr(10), ' ')}…"
            for d in hits
        ]
        return jsonify(answer=answer, sources=sources)

    except Exception:
        logger.exception("Error in /api/chat")
        return jsonify(error="Internal server error"), 500

@app.route("/")
def index():
    logger.debug("Serving index.html")
    return send_file("index.html")

if __name__ == "__main__":
    logger.info("Starting Flask on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000)
