import os, pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from pathlib import Path




def norm_str(v, lower=True):
    if pd.isna(v): 
        return None
    s = str(v).strip()
    return s.lower() if lower else s

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

data_file = os.getenv("DATA_FILE")

if not data_file:
    raise RuntimeError(f"DATA_FILE is not set. Put it in {ENV_PATH} like:\nDATA_FILE=./data/support_issues_dummy.csv")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
collection = os.getenv("QDRANT_COLLECTION","issues")

def row_to_text(r):
    return (
      f"Issue ID: {r['issue_ID']}\n"
      f"Issue Title: {r['issue_title']}\n"
      f"Summary: {r.get('summary','')}\n"
      f"Remediation: {r['remediation']}\n"
      f"Severity: {r.get('severity','')}\n"
      f"Category: {r.get('category','')}\n"
      f"System: {r.get('system','')}\n"
      f"Owner Team: {r.get('owner_team','')}\n"
      f"Status: {r.get('status','')}\n"
      f"Tags: {r.get('tags','')}"
    )

def main():
    p = Path(data_file).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"DATA_FILE points to a non-existent file: {p}")

    df = pd.read_excel(p) if p.suffix.lower() == ".xlsx" else pd.read_csv(p)
    docs = []
    for _, r in df.iterrows():
        meta = {
            "issue_ID": r["issue_ID"],
            "category": norm_str(r.get("category")),     # -> lowercase
            "severity": norm_str(r.get("severity")),
            "system":  norm_str(r.get("system"), lower=False),
            "owner_team": norm_str(r.get("owner_team")),
            "status": norm_str(r.get("status")),
        }


        docs.append(Document(
            page_content=row_to_text(r),
            metadata={
                "issue_ID": r["issue_ID"],
                "category": r.get("category"),
                "severity": r.get("severity"),
                "system": r.get("system"),
                "owner_team": r.get("owner_team"),
                "status": r.get("status"),
            }
        ))
    
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    
    Qdrant.from_documents(
        docs,
        emb,
        url=qdrant_url,
        api_key=qdrant_key,
        collection_name=collection,
    )
    print(f"✅ Upserted {len(docs)} docs into Qdrant collection '{collection}'")

if __name__ == "__main__":
    main()
