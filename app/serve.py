# app/serve.py

from __future__ import annotations

import os
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter as QFilter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ======================================================================================
# ENV + CONFIG
# ======================================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # validated in /ask
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # may be None for local qdrant
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "issues")

TOP_K = int(os.getenv("TOP_K", "8"))

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"]

# ======================================================================================
# QDRANT + VECTORSTORE
# ======================================================================================

# Embeddings for similarity search. Pass API key explicitly to avoid env pitfalls.
emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

# Qdrant client + VectorStore pointing to existing collection
qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vs = Qdrant(client=qclient, collection_name=QDRANT_COLLECTION, embeddings=emb)

# Higher-recall retriever (MMR) for general questions
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 40, "fetch_k": 200, "lambda_mult": 0.5},
)

# ======================================================================================
# FASTAPI APP
# ======================================================================================

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================================
# Pydantic Models
# ======================================================================================

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class RetrieveResponse(BaseModel):
    results: List[Dict[str, Any]]

# ======================================================================================
# Helpers: filters, formatting, synthesis
# ======================================================================================

def parse_filters(q: str) -> Dict[str, Any]:
    """
    Infer filters from the natural-language question.
    Produces keys:
      - severity__any: list[str]
      - tags__any: list[str]
      - assigned_person: str
      - listy: bool
    """
    qn = q.lower().replace("—", "-").replace("/", " ")
    f: Dict[str, Any] = {}

    # severity terms (multi)
    sev = [s for s in ["critical", "high", "medium", "low"] if s in qn]
    if sev:
        f["severity__any"] = sev

    # tags (multi): includes locations & product words we care about
    tags = []
    for t in [
        "oracle", "vpn", "split_tunnel",
        "pune", "chennai", "bengaluru", "hyderabad", "mumbai",
    ]:
        if t in qn:
            tags.append(t)
    if tags:
        f["tags__any"] = tags

    # assignee
    m = re.search(r"assigned\s+(?:to|person)\s+([a-z][a-z\.\-\s]+)", qn)
    if m:
        f["assigned_person"] = m.group(1).strip()

    # list intent
    f["listy"] = any(x in qn for x in ["list", "show", "ids only", "include issue id", "return ids"])
    return f

def to_qdrant_filter(f: Dict[str, Any]) -> Optional[QFilter]:
    must = []
    if "assigned_person" in f:
        must.append(FieldCondition(key="assigned_person", match=MatchValue(value=f["assigned_person"])))
    if "severity__any" in f:
        must.append(FieldCondition(key="severity", match=MatchAny(any=f["severity__any"])))
    if "tags__any" in f:
        must.append(FieldCondition(key="tags", match=MatchAny(any=f["tags__any"])))
    return QFilter(must=must) if must else None

def remediation_snippet(d, max_len: int = 220) -> Optional[str]:
    """Prefer metadata.remediation; fallback to parsing page_content 'Remediation:' line."""
    r = d.metadata.get("remediation")
    if not r:
        m = re.search(r"Remediation:\s*(.*)", d.page_content or "", flags=re.IGNORECASE | re.DOTALL)
        if m:
            chunk = m.group(1)
            # stop at next header-like label
            chunk = re.split(r"\n[A-Z][A-Za-z ]+:\s*", chunk)[0].strip()
            r = chunk
    if not r:
        return None
    r = re.sub(r"\s+", " ", str(r)).strip()
    return (r[: max_len - 1] + "…") if len(r) > max_len else r

def post_filter_docs(docs, f: Dict[str, Any]):
    """Apply strict client-side filtering to retrieved docs."""
    out = docs
    if "severity__any" in f:
        want = set(f["severity__any"])
        out = [d for d in out if (d.metadata.get("severity") in want)]
    if "tags__any" in f:
        want_tags = set(f["tags__any"])
        def has_tag(meta_tags):
            if isinstance(meta_tags, list):
                return bool(want_tags & set(meta_tags))
            if isinstance(meta_tags, str):
                return any(t in meta_tags for t in want_tags)
            return False
        out = [d for d in out if has_tag(d.metadata.get("tags"))]
    if "assigned_person" in f:
        out = [d for d in out if (d.metadata.get("assigned_person") == f["assigned_person"])]
    return out

def format_list_from_docs(docs, limit: int, include_remediation: bool = True) -> str:
    lines, seen = [], set()
    for d in docs:
        iid = d.metadata.get("issue_ID")
        if not iid or iid in seen:
            continue
        seen.add(iid)
        sys_ = d.metadata.get("system")
        stat_ = d.metadata.get("status")
        asg_ = d.metadata.get("assigned_person") or "—"
        base = f"- {iid} — {sys_} — {stat_} — {asg_}"
        if include_remediation:
            rem = remediation_snippet(d)
            if rem:
                base += f"\n  Remediation: {rem}"
        lines.append(base)
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "No matching tickets found."

def search_with_relaxation(question: str, k: int, f_in: Dict[str, Any]):
    """
    Progressive strategy:
      1) full filter
      2) drop tags
      3) drop severity
      4) each alone (if both were present)
      5) no filter (MMR retriever)
    Returns (docs, used_filter_dict)
    """
    trials: List[Dict[str, Any]] = []
    trials.append(f_in.copy())

    if "tags__any" in f_in:
        f2 = f_in.copy(); f2.pop("tags__any", None); trials.append(f2)
    if "severity__any" in f_in:
        f3 = f_in.copy(); f3.pop("severity__any", None); trials.append(f3)
    if "tags__any" in f_in and "severity__any" in f_in:
        trials.append({"tags__any": f_in["tags__any"]})
        trials.append({"severity__any": f_in["severity__any"]})
    trials.append({})  # no filter

    for ff in trials:
        qf = to_qdrant_filter(ff) if ff else None
        if qf:
            docs = vs.similarity_search(query=question, k=max(k, 20), filter=qf)
        else:
            docs = retriever.get_relevant_documents(question)
            cap = max(k, 20)
            docs = docs[:cap] if len(docs) > cap else docs

        if docs:
            return docs, ff
    return [], {}

def build_context(docs, max_chars: int = 9000) -> str:
    """Join unique docs into a compact context block."""
    seen, chunks, total = set(), [], 0
    for d in docs:
        iid = d.metadata.get("issue_ID")
        if iid in seen:
            continue
        seen.add(iid)
        header = (
            f"Issue ID: {iid}\n"
            f"Title: {d.metadata.get('issue_title') or ''}\n"
            f"Severity: {d.metadata.get('severity')}; "
            f"Category: {d.metadata.get('category')}; "
            f"System: {d.metadata.get('system')}; "
            f"Status: {d.metadata.get('status')}; "
            f"Assigned: {d.metadata.get('assigned_person') or '-'}\n"
        )
        rem = d.metadata.get("remediation")
        body = d.page_content or ""
        block = header + (f"Remediation: {rem}\n" if rem else "") + body
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(chunks) if chunks else ""

LLM_SYSTEM = (
    "You are a senior support engineer assistant. Use ONLY the provided CONTEXT.\n"
    "Write a concise, actionable answer. If the user asks to list items, give a short summary first, "
    "then include a clear step-by-step fix and verification steps. Explicitly reference Issue IDs in parentheses "
    "when you use them (e.g., '... (ISS-01041)'). Do NOT invent facts outside the context.\n"
)

def synthesize_answer(question: str, docs) -> str:
    """Call OpenAI to synthesize a natural-language answer from retrieved docs."""
    context = build_context(docs)
    if not context:
        return ""
    if not OPENAI_API_KEY:
        # Defer throwing until we actually try to call LLM
        raise RuntimeError("OPENAI_API_KEY not set on server")

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.1,
        max_tokens=800,
        api_key=OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", LLM_SYSTEM + "\nCONTEXT:\n{context}"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()

# ======================================================================================
# Routes
# ======================================================================================

@app.get("/health")
def health():
    return {"status": "ok", "collection": QDRANT_COLLECTION}

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    try:
        k = req.top_k or TOP_K
        docs = retriever.get_relevant_documents(req.query)   # IMPORTANT: req.query (not question)
        docs = docs[:k] if len(docs) > k else docs

        out = []
        for d in docs:
            out.append({
                "issue_ID": d.metadata.get("issue_ID"),
                "metadata": {
                    **{k: d.metadata.get(k) for k in [
                        "issue_ID", "category", "severity", "system",
                        "owner_team", "status", "assigned_person", "tags"
                    ]},
                    "_collection_name": QDRANT_COLLECTION,
                },
                "snippet": (d.page_content or "")[:300],
            })
        return RetrieveResponse(results=out)
    except Exception as e:
        print("ERROR in /retrieve:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"/retrieve failed: {e}")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        k = req.top_k or TOP_K
        f = parse_filters(req.question)

        # Progressive retrieval: full filter -> relax -> MMR
        docs, used_filter = search_with_relaxation(req.question, k, f)
        if not docs:
            return AskResponse(
                answer="No exact match found.\n\nRelated issues:\n- (none)\n\nNext steps:\nPlease refine filters or broaden the question.",
                citations=[]
            )

        # If listy or filters were used, try LLM synthesis first from the same docs
        if f.get("listy") or used_filter:
            # Prefer strict subset if available
            strict_docs = post_filter_docs(docs, f)
            show_docs = strict_docs if strict_docs else docs

            llm_answer = synthesize_answer(req.question, show_docs)
            if not llm_answer or "i don't know" in llm_answer.lower():
                # Helpful fallback: formatted list with remediation snippets
                preface = ""
                if (not strict_docs) and (("severity__any" in f) or ("tags__any" in f)):
                    want_bits = []
                    if "severity__any" in f:
                        want_bits.append("severity=" + "/".join(f["severity__any"]))
                    if "tags__any" in f:
                        want_bits.append("tags=" + "/".join(f["tags__any"]))
                    preface = f"No exact matches for ({', '.join(want_bits)}). Showing closest matches:\n"
                llm_answer = preface + format_list_from_docs(show_docs, k, include_remediation=True)

            # Citations from the docs we actually used
            seen, cits = set(), []
            for d in show_docs:
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
                    if len(cits) >= k:
                        break

            return AskResponse(answer=llm_answer, citations=cits)

        # Narrative path for general questions (not explicitly listy)
        llm_answer = synthesize_answer(req.question, docs)
        if not llm_answer or "i don't know" in llm_answer.lower():
            llm_answer = "Based on top matches:\n" + format_list_from_docs(docs, k, include_remediation=True)

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
                if len(cits) >= k:
                    break

        return AskResponse(answer=llm_answer, citations=cits)

    except Exception as e:
        print("ERROR in /ask:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"/ask failed: {e}")
