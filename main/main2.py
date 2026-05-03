import asyncio
import uuid
import time
from typing import Literal, Optional, Any
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from contextlib import asynccontextmanager

from app.database import get_db, init_db, Session as SessionModel, Message as MessageModel, AgentTrace
from main.agents import root_agent, InMemoryRunner

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="SROP API", version="1.0.0", lifespan=lifespan)

class HelixError(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    def __init__(self, detail: str = ""):
        self.detail = detail

class SessionNotFoundError(HelixError):
    status_code = 404
    error_code = "SESSION_NOT_FOUND"

class UpstreamTimeoutError(HelixError):
    status_code = 504
    error_code = "UPSTREAM_TIMEOUT"

@app.exception_handler(HelixError)
async def helix_error_handler(request: Request, exc: HelixError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://docs.helix.example/errors/{exc.error_code.lower()}",
            "title": exc.error_code,
            "status": exc.status_code,
            "detail": exc.detail,
        },
    )

class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: Literal["free", "pro", "enterprise"] = "free"

class CreateSessionResponse(BaseModel):
    session_id: str

class SendMessageRequest(BaseModel):
    content: str

class SendMessageResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str



@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/v1/sessions", response_model=CreateSessionResponse)
async def create_session(body: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    session_id = str(uuid.uuid4())
    session = SessionModel(
        session_id=session_id,
        user_id=body.user_id,
        plan_tier=body.plan_tier,
        state={"user_id": body.user_id, "plan_tier": body.plan_tier},
        turn_count=0
    )
    db.add(session)
    await db.commit()
    return CreateSessionResponse(session_id=session_id)

@app.post("/v1/chat/{session_id}", response_model=SendMessageResponse)
async def chat(session_id: str, body: SendMessageRequest, db: AsyncSession = Depends(get_db)):
    start_time = time.time()
    result = await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise SessionNotFoundError(f"Session {session_id} not found")

    # Add user message
    user_msg_id = str(uuid.uuid4())
    user_msg = MessageModel(
        message_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=body.content
    )
    db.add(user_msg)
    
    # Run Agent
    runner = InMemoryRunner(agent=root_agent)
    
    # Inject context
    context = {
        "user_id": session.user_id,
        "plan_tier": session.plan_tier,
        "turn_count": session.turn_count,
        "last_agent_used": session.last_agent_used
    }
    
    routed_to = "srop_root"
    tool_calls = []
    retrieved_chunk_ids = []
    final_text = ""
    
    try:
        # Wrap ADK run in timeout
        async def run_agent():
            evs = []
            async for event in runner.run_async(session_id, body.content, context):
                evs.append(event)
            return evs

        events = await asyncio.wait_for(run_agent(), timeout=30.0)
        
        for event in events:
            if event.type == "tool_call":
                tool_calls.append({
                    "tool_name": event.tool_name,
                    "args": event.tool_args,
                })
            elif event.type == "retrieved_chunks":
                retrieved_chunk_ids.extend(event.chunks)
            elif event.type == "final_response":
                routed_to = event.author
                final_text = event.content.parts[0].text
                
    except asyncio.TimeoutError:
        raise UpstreamTimeoutError("LLM did not respond in time")
    except Exception as e:
        # Log error in real app, wrap in HelixError
        raise HelixError(f"Agent error: {str(e)}")
        
    trace_id = str(uuid.uuid4())
    
    # Save Trace
    latency_ms = int((time.time() - start_time) * 1000)
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=retrieved_chunk_ids,
        latency_ms=latency_ms
    )
    db.add(trace)
    
    # Save Assistant Message
    asst_msg_id = str(uuid.uuid4())
    asst_msg = MessageModel(
        message_id=asst_msg_id,
        session_id=session_id,
        role="assistant",
        content=final_text
    )
    db.add(asst_msg)
    
    # Update Session State
    session.turn_count += 1
    session.last_agent_used = routed_to
    
    await db.commit()
    
    return SendMessageResponse(
        reply=final_text,
        routed_to=routed_to,
        trace_id=trace_id
    )

@app.get("/v1/traces/{trace_id}")
async def get_trace(trace_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AgentTrace).where(AgentTrace.trace_id == trace_id))
    trace = result.scalar_one_or_none()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
        
    return {
        "trace_id": trace.trace_id,
        "session_id": trace.session_id,
        "routed_to": trace.routed_to,
        "tool_calls": trace.tool_calls,
        "retrieved_chunk_ids": trace.retrieved_chunk_ids,
        "latency_ms": trace.latency_ms
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main.main2:app", host="0.0.0.0", port=8000, reload=True)
