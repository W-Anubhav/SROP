import json
import inspect
from typing import Callable, Any
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from main.ingest import search_docs

from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# --- Mock ADK ---
class AgentTool:
    def __init__(self, agent):
        self.agent = agent
        # Treat an agent like a callable tool that takes 'user_message'
        self.__name__ = agent.name
        self.__doc__ = f"Route to {agent.name}. {agent.instruction}"

    async def __call__(self, user_message: str, **kwargs):
        return await self.agent.run(user_message, **kwargs)

class ADKEvent:
    def __init__(self, type: str, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def is_final_response(self):
        return self.type == "final_response"

class LlmAgent:
    def __init__(self, name: str, model: str, instruction: str, tools: list):
        self.name = name
        self.model = "gpt-4o-mini" # override for speed and cost
        self.instruction = instruction
        self.tools = tools
        self._tool_map = {}
        self._openai_tools = []
        
        for tool in tools:
            if isinstance(tool, AgentTool):
                tool_name = tool.agent.name
                tool_desc = tool.__doc__
                params = {
                    "type": "object",
                    "properties": {
                        "user_message": {"type": "string", "description": "The message to pass to the agent"}
                    },
                    "required": ["user_message"]
                }
                func = tool
            else:
                tool_name = tool.__name__
                tool_desc = tool.__doc__ or "No description."
                # Extract simple params (assuming str/int annotations)
                sig = inspect.signature(tool)
                props = {}
                req = []
                for p_name, p in sig.parameters.items():
                    props[p_name] = {"type": "string"} # Simplify typing
                    if p.default == inspect.Parameter.empty:
                        req.append(p_name)
                params = {"type": "object", "properties": props, "required": req}
                func = tool
                
            self._openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": params
                }
            })
            self._tool_map[tool_name] = func

    async def run(self, user_message: str, history: list = None, yield_events=False) -> str | list:
        messages = [{"role": "system", "content": self.instruction}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        
        events = []
        
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if self._openai_tools:
            kwargs["tools"] = self._openai_tools
            
        res = await client.chat.completions.create(**kwargs)
        choice = res.choices[0]
        
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)
                events.append(ADKEvent("tool_call", tool_name=func_name, tool_args=args))
                
                func = self._tool_map[func_name]
                if inspect.iscoroutinefunction(func) or (hasattr(func, '__call__') and inspect.iscoroutinefunction(func.__call__)):
                    result = await func(**args)
                else:
                    result = func(**args)
                    
                if isinstance(result, list) and hasattr(result[0], 'chunk_id'):
                    # specific handling for chunks
                    events.append(ADKEvent("retrieved_chunks", chunks=[c.chunk_id for c in result]))
                    formatted_res = "\\n\\n".join([f"[{c.chunk_id}] {c.content}" for c in result])
                else:
                    formatted_res = str(result)
                    
                events.append(ADKEvent("tool_result", tool_name=func_name, result=formatted_res))
                
                # Recursive call with tool result
                messages.append(choice.message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": formatted_res
                })
                
                followup_res = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                final_text = followup_res.choices[0].message.content
                
                # Try to extract the true author if it was an AgentTool
                author = func_name if isinstance(self._tool_map[func_name], AgentTool) else self.name
                
                class Content:
                    def __init__(self, txt):
                        self.parts = [type('Part', (), {'text': txt})]
                        
                events.append(ADKEvent("final_response", author=author, content=Content(final_text)))
                if yield_events:
                    return events
                return final_text
                
        else:
            final_text = choice.message.content
            class Content:
                def __init__(self, txt):
                    self.parts = [type('Part', (), {'text': txt})]
            events.append(ADKEvent("final_response", author=self.name, content=Content(final_text)))
            if yield_events:
                return events
            return final_text


class InMemoryRunner:
    def __init__(self, agent):
        self.agent = agent
        
    async def run_async(self, session_id, user_message, state_dict=None):
        # We simulate the ADK runner.
        system_ctx = ""
        if state_dict:
            system_ctx = f"Current user context:\\n" + "\\n".join([f"- {k}: {v}" for k,v in state_dict.items()])
            
        # We pass it to history to give context
        history = []
        if system_ctx:
            history.append({"role": "system", "content": system_ctx})
            
        events = await self.agent.run(user_message, history=history, yield_events=True)
        for ev in events:
            yield ev

# --- Sub-agents & Tools ---

async def get_recent_builds(user_id: str, limit: str = "3") -> str:
    """Get the recent builds for the user. Call this to check failed or successful builds."""
    return f"Builds for {user_id}: 1. Failed (OOM), 2. Success, 3. Success."

async def get_account_status(user_id: str) -> str:
    """Get the account status for the user."""
    return f"Account {user_id} is active."

account_agent = LlmAgent(
    name="account",
    model="gemini-2.0-flash",
    instruction="You are the AccountAgent. Answer user account queries using your tools.",
    tools=[get_recent_builds, get_account_status]
)

knowledge_agent = LlmAgent(
    name="knowledge",
    model="gemini-2.0-flash",
    instruction="Answer product questions using the search_docs tool. Always cite chunk IDs like '[chunk_abc123]'.",
    tools=[search_docs]
)

root_agent = LlmAgent(
    name="srop_root",
    model="gemini-2.0-flash",
    instruction="""You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.
- HOW to do something, WHAT something is, docs/feature questions -> knowledge
- Their account, builds, status, usage -> account
- Greetings or off-topic -> respond directly
Always route to a tool when appropriate. Do not answer questions yourself if there is a tool.""",
    tools=[
        AgentTool(agent=knowledge_agent),
        AgentTool(agent=account_agent)
    ]
)
