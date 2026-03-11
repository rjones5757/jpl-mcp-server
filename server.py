"""
JPL Template Search — MCP Server

Connects Claude to the JPL template library stored in Pinecone.
Deployed on Railway. Claude Teams connects via streamable HTTP transport.

Tools:
  jpl_search_templates  — Semantic search by natural language query
  jpl_get_template      — Retrieve full metadata for a specific template
"""

import os
import json
import logging

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from openai import OpenAI
from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "jpl-templates"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialize API clients (module-level — persist across requests)
# ---------------------------------------------------------------------------

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(PINECONE_INDEX)

logger.info("API clients initialized (OpenAI + Pinecone index '%s')", PINECONE_INDEX)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

# Railway's proxy forwards the public domain as the Host header.
# The MCP SDK's DNS rebinding protection rejects any Host not on the
# allowed list. We add Railway's domain so the proxy can reach us.
# See: https://github.com/modelcontextprotocol/python-sdk/issues/1798

RAILWAY_HOST = os.environ.get(
    "RAILWAY_PUBLIC_DOMAIN",
    "web-production-acfd4.up.railway.app"
)

mcp = FastMCP(
    "jpl_template_mcp",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            f"{RAILWAY_HOST}:*",
            RAILWAY_HOST,
        ],
        allowed_origins=[
            "http://localhost:*",
            "https://localhost:*",
            f"https://{RAILWAY_HOST}",
            f"https://{RAILWAY_HOST}:*",
        ],
    ),
)


def _build_filter(practice_area: str, document_type: str) -> dict:
    """Build a Pinecone metadata filter dict from optional string params."""
    f = {}
    if practice_area:
        f["practice_area"] = practice_area
    if document_type:
        f["document_type"] = document_type
    return f


def _format_results(matches) -> list[dict]:
    """Convert Pinecone match objects to serializable dicts."""
    results = []
    for m in matches:
        results.append({
            "score": round(m.score, 4),
            "template_id": m.id,
            "metadata": m.metadata or {}
        })
    return results


# ---------------------------------------------------------------------------
# Tool: Search Templates
# ---------------------------------------------------------------------------

@mcp.tool(
    name="jpl_search_templates",
    annotations={
        "title": "Search JPL Template Library",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def jpl_search_templates(
    query: str,
    top_k: int = 10,
    practice_area: str = "",
    document_type: str = "",
) -> str:
    """Search the JPL template library by describing a legal scenario or
    document need in natural language.

    Returns matching templates ranked by relevance with metadata including
    practice area, document type, narrative summary, quality confidence,
    template tier, and similarity scores.

    Args:
        query: Natural language description of the legal scenario or
               document need — e.g. "quiet title petition for a tax sale
               property" or "I need to evict a commercial tenant for
               non-payment."
        top_k: Number of results to return (1–20, default 10).
        practice_area: Optional — limit results to a practice area
                       (e.g. "Foreclosure", "Quiet Title", "Evictions").
        document_type: Optional — limit results to a document type
                       (e.g. "Petition", "Motion", "Deed", "Lease").

    Returns:
        JSON object with result_count and an array of matching templates.
    """
    try:
        # 1. Generate embedding from the query
        embed_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_vector = embed_response.data[0].embedding

        # 2. Build optional metadata filter
        meta_filter = _build_filter(practice_area, document_type)

        # 3. Query Pinecone
        query_kwargs: dict = {
            "vector": query_vector,
            "top_k": min(max(top_k, 1), 20),
            "include_metadata": True,
        }
        if meta_filter:
            query_kwargs["filter"] = meta_filter

        results = index.query(**query_kwargs)

        # 4. Format and return
        formatted = _format_results(results.matches)

        if not formatted:
            return json.dumps({
                "message": "No matching templates found in the library.",
                "result_count": 0,
                "results": [],
            })

        return json.dumps({
            "result_count": len(formatted),
            "results": formatted,
        }, indent=2)

    except Exception as e:
        logger.error("Search error: %s", e, exc_info=True)
        return json.dumps({
            "error": f"Search failed: {e}",
            "suggestion": "Verify that the Pinecone index exists and API keys are valid.",
        })


# ---------------------------------------------------------------------------
# Tool: Get Template by ID
# ---------------------------------------------------------------------------

@mcp.tool(
    name="jpl_get_template",
    annotations={
        "title": "Get Template Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def jpl_get_template(template_id: str) -> str:
    """Retrieve the full metadata for a specific template by its Pinecone ID.

    Use this after jpl_search_templates to get complete details about a
    template — including its full narrative summary, template header,
    negative boundaries, companion documents, quality confidence, and all
    other metadata fields.

    Args:
        template_id: The template ID returned by jpl_search_templates.

    Returns:
        JSON object with the template's full metadata, or an error message
        if the ID is not found.
    """
    try:
        result = index.fetch(ids=[template_id])

        if template_id in result.vectors:
            vector_data = result.vectors[template_id]
            return json.dumps({
                "template_id": template_id,
                "metadata": vector_data.metadata or {},
            }, indent=2)

        return json.dumps({
            "error": f"Template '{template_id}' not found.",
            "suggestion": "Use jpl_search_templates to find valid template IDs.",
        })

    except Exception as e:
        logger.error("Fetch error: %s", e, exc_info=True)
        return json.dumps({
            "error": f"Failed to retrieve template: {e}",
        })


# ---------------------------------------------------------------------------
# ASGI app — used by uvicorn
# ---------------------------------------------------------------------------

app = mcp.streamable_http_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting JPL Template MCP server on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
