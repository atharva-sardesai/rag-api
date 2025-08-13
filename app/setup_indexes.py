import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

load_dotenv()  # reads .env in project root
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_API_KEY")
coll = os.getenv("QDRANT_COLLECTION", "issues")

client = QdrantClient(url=url, api_key=key)

fields = ["category","severity","system","owner_team","status","tags","assigned_person","issue_ID"  ]
for f in fields:
    try:
        client.create_payload_index(
            collection_name=coll,
            field_name=f,
            field_schema=PayloadSchemaType.KEYWORD
        )
        print(f"✅ Created index for '{f}' (KEYWORD)")
    except Exception as e:
        print(f"ℹ️  Skipping '{f}' -> {e}")

print("Done.")
