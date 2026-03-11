"""
JPL Template Search — MCP Server (v2)

Connects Claude to the JPL template library stored in Pinecone.
Deployed on Railway. Claude Teams connects via streamable HTTP transport.

Tools:
  jpl_search_templates  — Semantic search by natural language query
  jpl_get_template      — Retrieve full metadata for a specific template

Changes from v1:
  - jpl_get_template ID resolution: accepts base template_id without __primary suffix
  - Improved metadata presentation in search results
  - Better error messages and logging
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


# Fields to include in search result summaries (lightweight, most useful for Claude)
SEARCH_RESULT_FIELDS = [
    "template_id", "template_name", "document_type", "practice_area",
    "sub_practice_area", "template_tier", "quality_confidence",
    "narrative_summary", "jpl_doc_type", "service_modality",
    "negative_boundaries", "companion_documents", "vector_type",
]


def _format_results(matches) -> list[dict]:
    """Convert Pinecone match objects to serializable dicts.

    For search results, includes only the most useful metadata fields
    to keep response sizes manageable. Use jpl_get_template for full metadata.
    """
    results = []
    for m in matches:
        meta = m.metadata or {}

        # Build a focused result with the most useful fields
        result = {
            "score": round(m.score, 4),
            "vector_id": m.id,
            "template_id": meta.get("template_id", m.id.split("__")[0]),
        }

        # Include available search-relevant metadata fields
        for field in SEARCH_RESULT_FIELDS:
            if field in meta and field != "template_id":
                result[field] = meta[field]

        results.append(result)

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

    Results may include matches from multiple vector types per template
    (primary embeddings, HyPE hypothetical queries). The template_id field
    links all vectors belonging to the same template. Use jpl_get_template
    with the template_id to retrieve full metadata for any result.

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
        Each result includes a score, template_id, and key metadata fields.
        Use jpl_get_template to retrieve the complete metadata for any template.
    """
    try:
        embed_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_vector = embed_response.data[0].embedding

        meta_filter = _build_filter(practice_area, document_type)

        query_kwargs: dict = {
            "vector": query_vector,
            "top_k": min(max(top_k, 1), 20),
            "include_metadata": True,
        }
        if meta_filter:
            query_kwargs["filter"] = meta_filter

        results = index.query(**query_kwargs)

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

    Accepts either the base template_id (e.g.
    "curative-title-demand-letter-under-oklahoma-curative-act") or the
    full Pinecone vector ID with suffix (e.g.
    "curative-title-demand-letter-under-oklahoma-curative-act__primary").

    Args:
        template_id: The template ID — either the base ID from metadata
                     or the full vector ID from search results.

    Returns:
        JSON object with the template's full metadata, or an error message
        if the ID is not found.
    """
    try:
        # Build candidate IDs to try
        ids_to_try = [template_id]

        # If the ID doesn't already have a vector suffix, also try __primary
        if "__" not in template_id:
            ids_to_try.append(f"{template_id}__primary")

        for candidate_id in ids_to_try:
            result = index.fetch(ids=[candidate_id])

            if candidate_id in result.vectors:
                vector_data = result.vectors[candidate_id]
                return json.dumps({
                    "template_id": template_id.split("__")[0],
                    "vector_id": candidate_id,
                    "metadata": vector_data.metadata or {},
                }, indent=2)

        return json.dumps({
            "error": f"Template '{template_id}' not found.",
            "tried_ids": ids_to_try,
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
