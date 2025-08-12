# app/serve.py
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_qdrant import Qdrant 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
from fastapi import HTTPException
from qdrant_client.http.models import Filter as QFilter, FieldCondition, MatchValue

# --- env ---
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
COLL       = os.getenv("QDRANT_COLLECTION", "issues")
TOP_K      = int(os.getenv("TOP_K", "4"))
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")


# --- embeddings FIRST ---
emb = OpenAIEmbeddings(model="text-embedding-3-small")





# --- vector store / retriever ---
vs = Qdrant.from_existing_collection(
    embedding=emb,
    url=QDRANT_URL,
    api_key=QDRANT_KEY,
    collection_name=COLL,
)
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.5}
)


# --- LLM & prompt ---
SYSTEM = (
    "You are a support assistant. Use ONLY the provided context. "
    "If missing, say you don't know. Always cite Issue IDs used."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\nContext:\n{context}"),
    ("human", "{question}")
])
def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('issue_ID','?')}]\n{d.page_content}" for d in docs)
def chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=512)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

# --- FastAPI app ---
class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]

from pydantic import BaseModel
from typing import Any, Dict, List

class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None
    category: str | None = None  # optional metadata filter

class RetrieveResponse(BaseModel):
    results: List[Dict[str, Any]]




app = FastAPI(title="Issues RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.top_k:
        retriever.search_kwargs["k"] = req.top_k
    docs = retriever.invoke(req.question)
    if not docs:
        return AskResponse(answer="I don't know.", citations=[])
    answer = chain().invoke(req.question)
    seen, cits = set(), []
    for d in docs:
        iid = d.metadata.get("issue_ID")
        if iid and iid not in seen:
            seen.add(iid)
            cits.append({
                "issue_ID": iid,
                "category": d.metadata.get("category"),
                "system": d.metadata.get("system"),
                "severity": d.metadata.get("severity"),
            })
    return AskResponse(answer=answer, citations=cits)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    try:
        k = req.top_k or TOP_K

        # Build a typed Qdrant Filter (instead of a raw dict)
        q_filter = None
        if req.category:
            q_filter = QFilter(must=[
                FieldCondition(key="category", match=MatchValue(value=req.category))
            ])

        # Call the vectorstore directly so we can pass the filter safely
        docs = vs.similarity_search(query=req.query, k=k, filter=q_filter)

        return {"results": [
            {
                "issue_ID": d.metadata.get("issue_ID"),
                "metadata": d.metadata,
                "snippet": d.page_content[:500]
            } for d in docs
        ]}
    except Exception as e:
        # Return a useful error instead of a blank 500
        raise HTTPException(status_code=500, detail=f"/retrieve failed: {e}")
