import os
import glob
import re
import yaml
import hashlib
import chromadb
from openai import AsyncOpenAI


from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="helix_docs",
    metadata={"hnsw:space": "cosine"}
)

def make_chunk_id(file_path: str, chunk_index: int) -> str:
    raw = f"{file_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

def extract_frontmatter(text: str) -> tuple[dict, str]:
    match = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if not match:
        return {}, text
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        metadata = {}
    body = text[match.end():]
    return metadata, body

def chunk_fixed(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

async def ingest_docs(docs_dir: str = "docs"):
    print(f"Starting ingestion from {docs_dir}")
    md_files = glob.glob(os.path.join(docs_dir, "*.md"))
    
    all_chunks = []
    all_ids = []
    all_metadatas = []
    
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        meta, body = extract_frontmatter(content)
        # Convert any list values in meta to comma separated strings to comply with ChromaDB
        for k, v in meta.items():
            if isinstance(v, list):
                meta[k] = ", ".join(map(str, v))
                
        meta["source"] = os.path.basename(file_path)
        
        chunks = chunk_fixed(body)
        for i, chunk_text in enumerate(chunks):
            chunk_id = make_chunk_id(file_path, i)
            all_chunks.append(chunk_text)
            all_ids.append(chunk_id)
            all_metadatas.append(meta.copy())
            
    if not all_chunks:
        print("No chunks to ingest.")
        return

    print(f"Embedding {len(all_chunks)} chunks...")
    # Batch embeddings using OpenAI
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i+batch_size]
        res = await client.embeddings.create(input=batch_texts, model="text-embedding-3-small")
        batch_embeddings = [d.embedding for d in res.data]
        all_embeddings.extend(batch_embeddings)

    print(f"Upserting to ChromaDB...")
    collection.upsert(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metadatas
    )
    print("Ingestion complete.")

class DocChunk:
    def __init__(self, chunk_id: str, score: float, content: str, metadata: dict):
        self.chunk_id = chunk_id
        self.score = score
        self.content = content
        self.metadata = metadata

async def search_docs(query: str, k: int = 5) -> list[DocChunk]:
    res = await client.embeddings.create(input=[query], model="text-embedding-3-small")
    query_embedding = res.data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    chunks = []
    if not results["ids"] or not results["ids"][0]:
        return chunks
        
    for chunk_id, distance, doc, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        score = round(1.0 - distance, 4)
        chunks.append(DocChunk(chunk_id=chunk_id, score=score, content=doc, metadata=meta))
        
    return sorted(chunks, key=lambda c: c.score, reverse=True)

if __name__ == "__main__":
    import asyncio
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="docs", help="Path to docs folder")
    args = parser.parse_args()
    asyncio.run(ingest_docs(args.path))
