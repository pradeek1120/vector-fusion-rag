import uuid
import time
from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END
from openai import OpenAI

from config.settings import get_settings
from src.retriever import hybrid_retrieve
from src.reranker import rerank
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
openai_client = OpenAI(api_key=settings.openai_api_key)


# --- State definition ---

class RAGState(TypedDict):
    query: str
    route: str
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    answer: str
    sources: list[dict]
    trace_id: str
    timings: dict
    error: Optional[str]
    top_k: Optional[int]
    use_hyde: Optional[bool]


# --- System prompt ---

SYSTEM_PROMPT = """You are a precise, helpful assistant. Answer the user's question
based strictly on the provided context passages.

Rules:
- Use only information from the context. Do not hallucinate.
- If the context does not contain enough information, say: "I don't have enough
  information in my knowledge base to answer that."
- Cite specific parts of the context when possible.
- Be concise and direct. Avoid filler phrases.
- Format your response clearly using markdown if the answer is complex."""


# --- Graph nodes ---

def node_route_node(state: RAGState) -> RAGState:
    """
    Classify the query to decide retrieval strategy.
    Can be extended to route to SQL, web search, etc.
    """
    query = state["query"].lower()

    if any(w in query for w in ["today", "latest", "current", "news", "recent"]):
        route = "web"  # placeholder — extend with Tavily tool
    elif any(w in query for w in ["count", "total", "average", "sum", "how many"]):
        route = "sql"  # placeholder — extend with SQL agent
    else:
        route = "rag"

    logger.info("query_routed", route=route, query_preview=query[:60])
    return {**state, "route": route}


def node_retrieve(state: RAGState) -> RAGState:
    t0 = time.time()

    docs, timings = hybrid_retrieve(
        query=state["query"],
        top_k=state.get("top_k"),
        use_hyde=state.get("use_hyde"),
    )

    timings["retrieve_total_ms"] = round((time.time() - t0) * 1000)
    existing_timings = state.get("timings", {})

    return {
        **state,
        "retrieved_docs": docs,
        "timings": {**existing_timings, **timings},
    }


def node_rerank(state: RAGState) -> RAGState:
    candidates = state["retrieved_docs"][: settings.top_k_retrieve]
    reranked, rerank_ms = rerank(state["query"], candidates)

    existing_timings = state.get("timings", {})
    return {
        **state,
        "reranked_docs": reranked,
        "timings": {**existing_timings, "rerank_ms": rerank_ms},
    }


def node_generate(state: RAGState) -> RAGState:
    docs = state["reranked_docs"]

    if not docs:
        return {
            **state,
            "answer": "I could not find relevant information to answer your question.",
            "sources": [],
        }

    context_parts = []
    for i, doc in enumerate(docs):
        context_parts.append(
            f"[Source {i + 1}] ({doc.get('source', 'unknown')})\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {state['query']}",
        },
    ]

    t0 = time.time()
    resp = openai_client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    gen_ms = round((time.time() - t0) * 1000)

    answer = resp.choices[0].message.content.strip()
    sources = [
        {
            "text": d["text"][:300] + "..." if len(d["text"]) > 300 else d["text"],
            "source": d.get("source", ""),
            "rerank_score": round(d.get("rerank_score", 0.0), 4),
        }
        for d in docs
    ]

    existing_timings = state.get("timings", {})
    return {
        **state,
        "answer": answer,
        "sources": sources,
        "timings": {**existing_timings, "generate_ms": gen_ms},
    }


def node_fallback(state: RAGState) -> RAGState:
    """
    Handles non-RAG routes (web / sql) — stubs to extend.
    """
    route = state.get("route", "unknown")
    logger.warning("route_not_implemented", route=route)
    return {
        **state,
        "answer": (
            f"This query requires '{route}' retrieval which is not yet "
            "connected. Please rephrase or use a different query."
        ),
        "sources": [],
        "retrieved_docs": [],
        "reranked_docs": [],
    }


# --- Conditional routing ---

def should_use_rag(state: RAGState) -> Literal["retrieve", "fallback"]:
    return "retrieve" if state.get("route") == "rag" else "fallback"


# --- Build the graph ---

def build_rag_graph():
    g = StateGraph(RAGState)

    g.add_node("route_node", node_route_node)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rerank", node_rerank)
    g.add_node("generate", node_generate)
    g.add_node("fallback", node_fallback)

    g.set_entry_point("route_node")

    g.add_conditional_edges(
        "route_node",
        should_use_rag,
        {"retrieve": "retrieve", "fallback": "fallback"},
    )

    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    g.add_edge("fallback", END)

    return g.compile()


# Module-level compiled graph (shared across requests)
rag_graph = build_rag_graph()


# --- Public interface ---

def run_query(
    query: str,
    top_k: Optional[int] = None,
    use_hyde: Optional[bool] = None,
) -> dict:
    """
    Execute the full RAG pipeline.
    Returns the final state dict.
    """
    trace_id = str(uuid.uuid4())

    initial_state: RAGState = {
        "query": query,
        "route": "rag",
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "sources": [],
        "trace_id": trace_id,
        "timings": {},
        "error": None,
        "top_k": top_k,
        "use_hyde": use_hyde,
    }

    try:
        t0 = time.time()
        final_state = rag_graph.invoke(initial_state)
        final_state["timings"]["total_ms"] = round((time.time() - t0) * 1000)
        return final_state
    except Exception as e:
        logger.error("graph_error", trace_id=trace_id, error=str(e))
        return {
            **initial_state,
            "answer": "An internal error occurred. Please try again.",
            "error": str(e),
        }
