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
    qn = q.lower().replace("—","-").replace("/", " ")
    f = {}
    # severities (allow multi)
    sev = [s for s in ["critical","high","medium","low"] if s in qn]
    if sev: f["severity__any"] = sev
    # tags from question
    tags = []
    for t in ["oracle","vpn","split_tunnel","pune","chennai","bengaluru","hyderabad","mumbai"]:
        if t in qn: tags.append(t)
    if tags: f["tags__any"] = tags
    # assignee
    m = re.search(r"assigned\s+(?:to|person)\s+([a-z][a-z\.\-\s]+)", qn)
    if m: f["assigned_person"] = m.group(1).strip()
    # list intent
    f["listy"] = any(x in qn for x in ["list","show","ids only","include issue id","return ids"])
    return f

def to_qdrant_filter(f: dict):
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

def search_with_relaxation(question: str, k: int, f_in: dict):
    trials = []
    # 1) full
    trials.append(f_in.copy())
    # 2) drop tags
    if "tags__any" in f_in:
        f2 = f_in.copy(); f2.pop("tags__any", None); trials.append(f2)
    # 3) drop severity
    if "severity__any" in f_in:
        f3 = f_in.copy(); f3.pop("severity__any", None); trials.append(f3)
    # 4) each alone (if both present)
    if "tags__any" in f_in and "severity__any" in f_in:
        trials.append({"tags__any": f_in["tags__any"]})
        trials.append({"severity__any": f_in["severity__any"]})
    # 5) no filter (MMR)
    trials.append({})

    for ff in trials:
        qf = to_qdrant_filter(ff) if ff else None
        if qf:
            docs = vs.similarity_search(query=question, k=max(k, 20), filter=qf)
        else:
            docs = retriever.get_relevant_documents(question)
            docs = docs[:max(k, 20)] if len(docs) > max(k, 20) else docs
        if docs:
            return docs, ff
    return [], {}

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
    "If context is thin, respond with 'Based on top matches' and list the Issue IDs used. "
    "Do not answer 'I don't know'."
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

        # Try full filter → relax → MMR
        docs, used_filter = search_with_relaxation(req.question, k, f)

        if not docs:
            return AskResponse(
                answer="No exact match found.\n\nRelated issues:\n- (none)\n\nNext steps:\nPlease refine filters or broaden the question.",
                citations=[]
            )

        # If listy or any filter used, return a list immediately
        if f.get("listy") or used_filter:
            ans = format_list_from_docs(docs, k)
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

        # Narrative RAG (non-list questions)
        context = format_docs_unique(docs)
        OPENAI_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_KEY:
            raise RuntimeError("OPENAI_API_KEY not set on server")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=600, api_key=OPENAI_KEY)
        rag = prompt | llm | StrOutputParser()
        answer = rag.invoke({"context": context, "question": req.question}).strip()

        # Never return "I don't know"
        if not answer or "i don't know" in answer.lower():
            answer = "Based on top matches:\n" + format_list_from_docs(docs, k)

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
        docs = retriever.get_relevant_documents(req.query)  # not req.question

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
