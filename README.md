# Helix SROP — Anubhav Maurya

## Setup

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Setup your OpenAI API key as an environment variable (or hardcode it for local testing if preferred)
# export OPENAI_API_KEY="sk-proj-..."

# 3. Ingest documents and initialize the ChromaDB vector store
python main/ingest.py --path docs/

# 4. Start the FastAPI application
uvicorn main.main2:app --reload
```

## Quick Test

```bash
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | jq -r .session_id)

curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | jq .
```

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌───────────────────────────────────────┐
│           FastAPI Server              │
│  1. Load `SessionModel` from SQLite   │
│  2. Build Context Dict (plan, turns)  │
│  3. Run ADK InMemoryRunner            │
│  4. Await `final_response`            │
│  5. Save `AgentTrace` & updated State │
└────────────┬──────────────────────────┘
             │ AgentTool Routing Loop
      ┌──────┴──────┐
      ▼             ▼
KnowledgeAgent   AccountAgent
(OpenAI + RAG)   (OpenAI + DB Tool)
      │
      ▼
ChromaDB Vector Store
```

## Design Decisions

### State persistence (which pattern and why)
I used **Pattern 3: Store only SessionState in DB, pass as system context**. 
Instead of trying to serialize and persist the entire complex history of ADK `Message` objects into the database, it's significantly more reliable to store the exact scalar metadata variables (`plan_tier`, `turn_count`, `last_agent_used`) as explicit SQLAlchemy columns/JSON fields. When a new turn starts, this explicit state is queried securely from the persistent SQLite database and injected into the dynamic system instruction of the root orchestrator. This completely guarantees state survival across `uvicorn` restarts while keeping database rows extremely lightweight.

### Chunking strategy
I used **Fixed-size chunking (with overlap)**. 
I implemented character-based chunking spanning 512 characters with an overlapping window of 64 characters. While heading-aware chunking is structurally pleasing, fixed-size with a healthy overlap guarantees that the mathematical density of the vector embeddings remains uniformly balanced across all chunks, while the overlap ensures that semantic meaning isn't completely severed if a sentence breaks.

### Vector store choice
I chose **ChromaDB**. 
Chroma provides built-in persistence to a local directory (`./chroma_db/`) immediately out of the box without requiring the user to run external Docker containers. It seamlessly handles cosine-similarity indexing out of the box.

## Known Limitations

- **Google-ADK Availability:** Since the `google-adk` package is an internal mock library and unavailable via `pip`, I built a robust shim interface in `main/agents.py` using pure OpenAI function calling that strictly mimics the intended `AgentTool`, `LlmAgent`, and `InMemoryRunner` architectures. 
- **Synchronous Vector Calls:** While the database models are 100% `aiosqlite` async, the `chromadb` queries are executed synchronously in the event loop, which might marginally degrade concurrent performance under highly intense throughput loads.

## What I'd Do With More Time

- **Streaming SSE (E3):** I would implement Python asynchronous generators to yield `AgentTool` thought-process traces and text chunks securely over a `StreamingResponse` allowing the frontend UI to display real-time typing indicators.
- **Reranking (E4):** Add an LLM-as-a-judge reranking step prior to surfacing the `search_docs` results to eliminate low-context chunks that marginally passed the cosine similarity threshold.

## Time Spent

| Phase | Time |
|-------|------|
| Setup + DB + FastAPI boilerplate | 45 min |
| RAG ingest + search_docs | 35 min |
| ADK agents | 50 min |
| pipeline.py + state persistence | 40 min |
| Tests | 25 min |
| README | 15 min |
| **Total** | **3h 30m** |

## Extensions Completed

- [ ] E1: Idempotency
- [ ] E2: Escalation agent
- [ ] E3: Streaming SSE
- [ ] E4: Reranking
- [ ] E5: Guardrails
- [ ] E6: Docker
- [ ] E7: Eval harness
