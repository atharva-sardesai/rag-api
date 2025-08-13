# app/serve.py
import os, traceback
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
import re
from qdrant_client.http.models import Filter as QFilter, FieldCondition, MatchValue, MatchAny

def parse_filters(q: str) -> dict:
    ql = q.lower()
    out = {}

    # severity (allow multiple)
    sev = [s for s in ["critical","high","medium","low"] if s in ql]
    if sev: out["severity__any"] = sev

    # site/product hints into tags
    tags = []
    for t in ["pune","chennai","bengaluru","hyderabad","mumbai","oracle","vpn","split_tunnel"]:
        if t in ql: tags.append(t)
    if tags: out["tags__any"] = tags

    # assignee
    m = re.search(r"assigned (?:to|person)\s+([a-z][a-z\.\-\s]+)", ql)
    if m: out["assigned_person"] = m.group(1).strip()

    # list intent
    out["listy"] = any(x in ql for x in ["list","show","ids only","include issue id","return ids"])
    return out


def to_qdrant_filter(f: dict) -> QFilter | None:
    must = []
    if "assigned_person" in f:
        must.append(FieldCondition(key="assigned_person", match=MatchValue(value=f["assigned_person"])))
    if "severity__any" in f:
        must.append(FieldCondition(key="severity", match=MatchAny(any=f["severity__any"])))
    if "tags__any" in f:
        must.append(FieldCondition(key="tags", match=MatchAny(any=f["tags__any"])))
    return QFilter(must=must) if must else None

def format_list_from_docs(docs, limit):
    out, seen = [], set()
    for d in docs:
        iid = d.metadata.get("issue_ID")
        if not iid or iid in seen: continue
        seen.add(iid)
        out.append(f"- {iid} — {d.metadata.get('system')} — {d.metadata.get('status')} — {d.metadata.get('assigned_person')}")
        if len(out) >= limit: break
    return "\n".join(out) if out else "No matching tickets found."

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
    "If context is thin, return a best-effort answer prefixed with 'Based on top matches', "
    "and list the Issue IDs used. Never answer 'I don't know'."
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


def format_docs_unique(docs):
    seen, chunks = set(), []
    for d in docs:
        iid = d.metadata.get("issue_ID")
        if iid in seen:
            continue
        seen.add(iid)
        chunks.append(f"[{iid}]\n{d.page_content}")
    return "\n\n".join(chunks)

app = FastAPI(title="Issues RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        k = req.top_k or TOP_K
        f = parse_filters(req.question)
        q_filter = to_qdrant_filter(f)

        # Higher recall for listy questions
        search_kwargs = {"k": max(k, 20)} if f.get("listy") else {"k": k}

        # Use filter when present
        if q_filter:
            docs = vs.similarity_search(query=req.question, filter=q_filter, **search_kwargs)
        else:
            # you already have an MMR retriever defined; use it for general questions
            docs = retriever.get_relevant_documents(req.question)
            docs = docs[:k] if len(docs) > k else docs

        # If it looks like a list/lookup, return a list from the docs directly
        if f.get("listy") or q_filter:
            if docs:
                ans = format_list_from_docs(docs, k)
                # citations
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
                            "status": d.metadata.get("status"),
                            "assigned_person": d.metadata.get("assigned_person"),
                        })
                        if len(cits) >= k: break
                return AskResponse(answer=ans, citations=cits)

        if not docs:
            return AskResponse(
                answer="No exact match found.\n\nRelated issues:\n- (none)\n\nNext steps:\nPlease refine filters or broaden the question.",
                citations=[]
            )

        # Narrative path (RAG)
        context = format_docs_unique(docs)
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_KEY:
            raise RuntimeError("OPENAI_API_KEY not set on server")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=600, api_key=OPENAI_KEY)
        rag = prompt | llm | StrOutputParser()
        answer = rag.invoke({"context": context, "question": req.question}).strip()

        # Guardrail: never return "I don't know"
        if not answer or "i don't know" in answer.lower():
            ans = "Based on top matches:\n" + format_list_from_docs(docs, k)
            answer = ans

        # Citations
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
                    "status": d.metadata.get("status"),
                    "assigned_person": d.metadata.get("assigned_person"),
                })
                if len(cits) >= k: break

        return AskResponse(answer=answer, citations=cits)

    except Exception as e:
        print("ERROR in /ask:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"/ask failed: {e}")


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
        docs = retriever.get_relevant_documents(req.question)
        docs = docs[:k] if k and len(docs) > k else docs

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
