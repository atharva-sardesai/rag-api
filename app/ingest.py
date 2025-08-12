import os, pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

load_dotenv()
data_file = os.getenv("DATA_FILE")
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
    df = pd.read_excel(data_file) if data_file.endswith(".xlsx") else pd.read_csv(data_file)
    docs = []
    for _, r in df.iterrows():
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
