"""
JPL Template System — MCP Server (v4)

Connects Claude to the JPL template library (Pinecone) and the JPL
formatting macro service (Azure VM).

Deployed on Railway. Claude Teams connects via streamable HTTP transport.

Tools:
  jpl_search_templates    — Semantic search with deduplication, metadata
                            hydration, and enriched evaluation metadata
  jpl_get_template        — Retrieve full metadata for a specific template
  apply_jpl_formatting    — Run the JPL formatting macro on a .docx file
                            via the Azure VM macro service

Changes from v3:
  - jpl_search_templates now DEDUPLICATES by template_id — each template
    appears once regardless of how many vectors matched, with best score
    and a match_types array showing which vector types hit
  - HyPE-only matches are HYDRATED with the primary vector's full metadata
    via Pinecone fetch-by-ID (fast batch operation)
  - Search response includes ENRICHED evaluation metadata (narrative_summary,
    negative_boundaries, quality_confidence, distinctiveness_summary,
    factual_scenario, companion_documents, etc.) enabling Claude-side
    reranking without separate jpl_get_template calls
  - Default top_k increased to 20 to support the two-stage
    retrieve-and-rerank architecture
  - New parameter: include_full_metadata (bool) for complete metadata
  - jpl_get_template and apply_jpl_formatting unchanged from v3

Dependencies (requirements.txt):
  - mcp>=1.0.0
  - openai>=1.0.0
  - pinecone>=3.0.0
  - uvicorn>=0.24.0
  - httpx>=0.25.0
  - box-sdk-gen
"""

import os
import io
import json
import logging
import tempfile
from pathlib import Path

import httpx
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

# Macro API on the Azure VM
VM_MACRO_URL = os.environ.get("VM_MACRO_URL", "")       # e.g. http://<VM-IP>:8080/format
VM_MACRO_API_KEY = os.environ.get("VM_MACRO_API_KEY", "")

# Box JWT config — stored as a JSON string in the Railway env var
BOX_JWT_CONFIG = os.environ.get("BOX_JWT_CONFIG", "")

# Box folder for macro output
BOX_SKILLS_FOLDER = "365569381705"

# Timeout for the macro API call
MACRO_API_TIMEOUT = 120  # seconds

# ---------------------------------------------------------------------------
# Evaluation metadata fields — the subset Claude needs for intelligent triage.
# These are returned by default in search results. Use include_full_metadata=true
# for the complete 43+ field set.
# ---------------------------------------------------------------------------

EVALUATION_FIELDS = [
    # Identity
    "template_id",
    "template_tier",
    "document_type",
    "practice_area",
    "sub_practice_area",
    "jpl_doc_type",
    "service_modality",
    # For scenario matching
    "narrative_summary",
    "factual_scenario",
    "party_posture",
    "complexity",
    # For negative boundary triage
    "negative_boundaries",
    # For differentiation
    "distinctiveness_summary",
    "quality_confidence",
    # For companion/set retrieval
    "companion_documents",
    # For presentation
    "template_name",
    "key_protective_provisions",
    "advocacy_position",
    "jurisdiction",
    "property_type",
]

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
# Box client (lazy init — only created when the formatting tool is called)
# ---------------------------------------------------------------------------

_box_client = None


def _get_box_client():
    """Create or return the cached Box client.

    Uses the BOX_JWT_CONFIG environment variable, which contains the full
    JWT config JSON as a string (no file path needed on Railway).
    """
    global _box_client
    if _box_client is not None:
        return _box_client

    if not BOX_JWT_CONFIG:
        raise RuntimeError(
            "BOX_JWT_CONFIG environment variable is not set. "
            "Cannot connect to Box for file operations."
        )

    from box_sdk_gen import BoxClient, BoxJWTAuth, JWTConfig

    jwt_config = JWTConfig.from_config_json_string(BOX_JWT_CONFIG)
    auth = BoxJWTAuth(config=jwt_config)
    _box_client = BoxClient(auth=auth)

    user = _box_client.users.get_user_me()
    logger.info("Box client initialized (authenticated as: %s)", user.name)
    return _box_client


def _box_download_file(file_id: str) -> tuple[bytes, str]:
    """Download a file from Box. Returns (content_bytes, filename)."""
    client = _get_box_client()
    file_info = client.files.get_file_by_id(file_id)
    content = client.downloads.download_file(file_id).read()
    return content, file_info.name


def _box_upload_file(content: bytes, filename: str, folder_id: str) -> str:
    """Upload a file to Box. Returns the new file's ID."""
    from box_sdk_gen import UploadFileAttributes, UploadFileAttributesParentField

    client = _get_box_client()
    attrs = UploadFileAttributes(
        name=filename,
        parent=UploadFileAttributesParentField(id=folder_id),
    )
    uploaded = client.uploads.upload_file(attrs, io.BytesIO(content))
    return uploaded.entries[0].id


def _box_find_or_create_subfolder(parent_folder_id: str, folder_name: str) -> str:
    """Find a subfolder by name, or create it. Returns the folder ID."""
    from box_sdk_gen import CreateFolderParent

    client = _get_box_client()
    items = client.folders.get_folder_items(parent_folder_id)
    for entry in items.entries:
        if entry.type == "folder" and entry.name == folder_name:
            return entry.id

    # Not found — create it
    folder = client.folders.create_folder(
        name=folder_name,
        parent=CreateFolderParent(id=parent_folder_id),
    )
    logger.info("Created Box folder '%s' (ID: %s)", folder_name, folder.id)
    return folder.id


# ---------------------------------------------------------------------------
# Search helper functions — deduplication, hydration, metadata extraction
# ---------------------------------------------------------------------------

def _deduplicate_results(matches) -> list[dict]:
    """Deduplicate Pinecone results by template_id.

    When the same template matches on multiple vector types (primary,
    HyPE), collapse into a single entry with:
    - The best (highest) score across all matching vectors
    - A list of which vector types matched (match_types)
    - The richest metadata available (primary > HyPE)

    Returns a list sorted by best score descending.
    """
    by_template: dict[str, dict] = {}

    for m in matches:
        meta = m.metadata or {}
        template_id = meta.get("template_id", m.id.split("__")[0])
        vector_type = meta.get("vector_type", "unknown")
        score = m.score

        if template_id not in by_template:
            by_template[template_id] = {
                "template_id": template_id,
                "best_score": score,
                "match_types": [vector_type],
                "metadata": dict(meta),
                "has_primary_metadata": (vector_type == "primary"),
            }
        else:
            entry = by_template[template_id]
            if vector_type not in entry["match_types"]:
                entry["match_types"].append(vector_type)
            if score > entry["best_score"]:
                entry["best_score"] = score
            # Prefer primary metadata over HyPE (primary carries full 43+ fields)
            if vector_type == "primary" and not entry["has_primary_metadata"]:
                entry["metadata"] = dict(meta)
                entry["has_primary_metadata"] = True

    return sorted(by_template.values(), key=lambda x: x["best_score"], reverse=True)


def _hydrate_metadata(deduplicated: list[dict]) -> list[dict]:
    """Fetch full primary-vector metadata for templates matched only via HyPE.

    HyPE vectors carry minimal metadata (6 fields). When a template was
    found only through HyPE matches, we fetch the primary vector's metadata
    via Pinecone's fetch-by-ID (fast batch operation) to give Claude the
    narrative_summary, negative_boundaries, etc. it needs for evaluation.
    """
    needs_hydration = [
        entry for entry in deduplicated
        if not entry["has_primary_metadata"]
    ]

    if not needs_hydration:
        return deduplicated

    primary_ids = [f"{entry['template_id']}__primary" for entry in needs_hydration]

    try:
        fetch_response = index.fetch(ids=primary_ids)
        fetched_vectors = fetch_response.vectors or {}

        for entry in needs_hydration:
            primary_id = f"{entry['template_id']}__primary"
            if primary_id in fetched_vectors:
                primary_meta = fetched_vectors[primary_id].metadata or {}
                entry["metadata"] = dict(primary_meta)
                entry["has_primary_metadata"] = True
                logger.info("Hydrated metadata for: %s", entry["template_id"])
            else:
                logger.warning("Primary vector not found for: %s", entry["template_id"])

    except Exception as e:
        logger.error("Metadata hydration failed: %s", e)
        # Non-fatal — results still have stub metadata from HyPE vectors

    return deduplicated


def _extract_evaluation_metadata(metadata: dict) -> dict:
    """Extract the evaluation subset of metadata fields.

    Returns only the ~20 fields Claude needs for intelligent triage.
    Filters out empty/null values to keep the response compact.
    """
    result = {}
    for field in EVALUATION_FIELDS:
        value = metadata.get(field)
        if value is not None and value != "" and value != []:
            result[field] = value
    return result


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


# ---------------------------------------------------------------------------
# Tool: Search Templates (v4 — dedup + hydration + enriched metadata)
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
    top_k: int = 20,
    practice_area: str = "",
    document_type: str = "",
    include_full_metadata: bool = False,
) -> str:
    """Search the JPL template library by describing a legal scenario or
    document need in natural language.

    Returns matching templates ranked by relevance, DEDUPLICATED by template
    (each template appears once regardless of how many vectors matched), with
    ENRICHED metadata for intelligent evaluation — including narrative summary,
    negative boundaries, quality confidence, distinctiveness summary, factual
    scenario, companion documents, and other fields Claude needs to make
    informed recommendations.

    Results include match_types showing which vector types matched (primary,
    hype) — useful context for understanding why a template surfaced.

    The default top_k of 20 supports broad retrieval for Claude-side reranking.
    Claude should evaluate all results using the metadata, eliminate poor
    matches via negative boundary triage, and present the best 1-3 to the user.

    Args:
        query: Natural language description of the legal scenario or
               document need — e.g. "quiet title petition for a tax sale
               property" or "I need to evict a commercial tenant for
               non-payment."
        top_k: Number of raw Pinecone results before deduplication (1-50,
               default 20). After deduplication, unique template count will
               be lower. Increase for very broad searches.
        practice_area: Optional — limit results to a practice area
                       (e.g. "Foreclosure", "Quiet Title", "Evictions").
        document_type: Optional — limit results to a document type
                       (e.g. "Petition", "Motion", "Deed", "Lease").
        include_full_metadata: If true, return ALL metadata fields for every
                               result (equivalent to calling jpl_get_template
                               on each). Default false returns the evaluation
                               subset optimized for triage.

    Returns:
        JSON object with unique_template_count and an array of deduplicated,
        enriched template results. Use jpl_get_template to retrieve the
        complete metadata for any finalist template.
    """
    try:
        # Generate embedding
        embed_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_vector = embed_response.data[0].embedding

        # Build filter and query Pinecone
        meta_filter = _build_filter(practice_area, document_type)
        query_kwargs: dict = {
            "vector": query_vector,
            "top_k": min(max(top_k, 1), 50),
            "include_metadata": True,
        }
        if meta_filter:
            query_kwargs["filter"] = meta_filter

        results = index.query(**query_kwargs)
        raw_matches = results.matches or []

        logger.info("Search: '%s' → %d raw matches", query[:80], len(raw_matches))

        if not raw_matches:
            return json.dumps({
                "unique_template_count": 0,
                "raw_match_count": 0,
                "results": [],
                "message": "No templates found matching your query.",
            })

        # Deduplicate by template_id
        deduplicated = _deduplicate_results(raw_matches)

        # Hydrate HyPE-only matches with primary vector metadata
        deduplicated = _hydrate_metadata(deduplicated)

        logger.info("After dedup: %d unique templates from %d raw matches",
                     len(deduplicated), len(raw_matches))

        # Build response with evaluation or full metadata
        formatted_results = []
        for entry in deduplicated:
            metadata = entry["metadata"]

            if include_full_metadata:
                result_metadata = {
                    k: v for k, v in metadata.items()
                    if v is not None and v != "" and v != []
                }
            else:
                result_metadata = _extract_evaluation_metadata(metadata)

            result = {
                "score": round(entry["best_score"], 4),
                "template_id": entry["template_id"],
                "match_types": entry["match_types"],
                **result_metadata,
            }
            formatted_results.append(result)

        return json.dumps({
            "unique_template_count": len(formatted_results),
            "raw_match_count": len(raw_matches),
            "results": formatted_results,
        })

    except Exception as e:
        logger.error("Search error: %s", e, exc_info=True)
        return json.dumps({
            "error": f"Search failed: {e}",
            "suggestion": "Verify that the Pinecone index exists and API keys are valid.",
        })


# ---------------------------------------------------------------------------
# Tool: Get Template by ID (unchanged from v3)
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
# Tool: Apply JPL Formatting (macro-as-a-service)
# ---------------------------------------------------------------------------

@mcp.tool(
    name="apply_jpl_formatting",
    annotations={
        "title": "Apply JPL Formatting Macro",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def apply_jpl_formatting(
    file_content_base64: str = "",
    filename: str = "document.docx",
    box_file_id: str = "",
    output_folder_id: str = "",
) -> str:
    """Apply the JPL formatting macro to a .docx file.

    This tool bridges Claude to the JPL formatting macro running on the
    Azure VM. It sends the file to the macro service for formatting
    (margins, spacing, fonts, borders, etc.) and returns the result.

    TWO MODES — use one:

    Mode 1 (PREFERRED for chat sessions): Pass the .docx file content
    directly as base64. The macro runs and the formatted file is returned
    as base64. No Box involvement. Use this when Claude has just built
    a .docx locally.
      - file_content_base64: base64-encoded .docx file content (required)
      - filename: the filename (used for logging and the returned file)

    Mode 2 (for pipeline/Box workflows): Pass a Box file ID. The file
    is downloaded from Box, macro runs, result is uploaded back to Box.
      - box_file_id: Box file ID of the .docx (required)
      - output_folder_id: optional Box folder for the result

    Args:
        file_content_base64: Base64-encoded .docx file content. Use this
                             mode when Claude has built the file locally.
                             Mutually exclusive with box_file_id.
        filename: Filename for the document (default: document.docx).
                  Used for logging and the returned filename.
        box_file_id: Box file ID of the .docx to format. Use this mode
                     when the file is already in Box.
        output_folder_id: (Box mode only) Box folder ID for the result.
                          Defaults to "Macro Output" folder.

    Returns:
        Mode 1: JSON with status, formatted_file_base64, and filename.
        Mode 2: JSON with status, formatted_file_id, and filename.
        On error: JSON with error details and fallback instructions.
    """
    import base64
    from datetime import datetime as dt

    # --- Validate configuration ---
    if not VM_MACRO_URL:
        return json.dumps({
            "error": "Macro service not configured.",
            "detail": "VM_MACRO_URL environment variable is not set on the MCP server.",
            "fallback": "Instruct the user to download the .docx and run the JPL Format macro manually in Word.",
        })

    if not file_content_base64 and not box_file_id:
        return json.dumps({
            "error": "No file provided. Supply either file_content_base64 or box_file_id.",
        })

    try:
        # --- Step 1: Get the file content ---
        if file_content_base64:
            # Mode 1: Direct base64 content
            mode = "direct"
            try:
                file_content = base64.b64decode(file_content_base64)
            except Exception as e:
                return json.dumps({
                    "error": f"Invalid base64 content: {e}",
                })
            if not filename.lower().endswith(".docx"):
                filename = filename + ".docx"
            logger.info("apply_jpl_formatting (direct mode): %s (%s bytes)",
                        filename, f"{len(file_content):,}")
        else:
            # Mode 2: Download from Box
            mode = "box"
            logger.info("apply_jpl_formatting (box mode): downloading Box file %s", box_file_id)
            file_content, filename = _box_download_file(box_file_id)
            if not filename.lower().endswith(".docx"):
                return json.dumps({
                    "error": f"File '{filename}' is not a .docx file.",
                    "detail": "The macro service only accepts .docx files.",
                })
            logger.info("  Downloaded: %s (%s bytes)", filename, f"{len(file_content):,}")

        # --- Step 2: Send to the VM macro API ---
        logger.info("  Sending to macro service: %s", VM_MACRO_URL)

        headers = {}
        if VM_MACRO_API_KEY:
            headers["Authorization"] = f"Bearer {VM_MACRO_API_KEY}"

        async with httpx.AsyncClient(timeout=MACRO_API_TIMEOUT) as client:
            response = await client.post(
                VM_MACRO_URL,
                files={"file": (filename, file_content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                headers=headers,
            )

        if response.status_code != 200:
            error_detail = response.text[:500] if response.text else "No response body"
            logger.error("  Macro service returned %d: %s", response.status_code, error_detail)
            return json.dumps({
                "error": f"Macro service returned HTTP {response.status_code}.",
                "detail": error_detail,
                "fallback": "Instruct the user to download the .docx and run the JPL Format macro manually in Word.",
            })

        formatted_content = response.content
        logger.info("  Macro service returned formatted file (%s bytes)", f"{len(formatted_content):,}")

        # --- Step 3: Return the result ---
        if mode == "direct":
            # Mode 1: Return the formatted file as base64
            formatted_base64 = base64.b64encode(formatted_content).decode("ascii")
            logger.info("  Returning formatted file as base64 (%s chars)", f"{len(formatted_base64):,}")

            return json.dumps({
                "status": "success",
                "mode": "direct",
                "formatted_file_base64": formatted_base64,
                "filename": filename,
            })

        else:
            # Mode 2: Upload to Box
            dest_folder = output_folder_id
            if not dest_folder:
                dest_folder = _box_find_or_create_subfolder(BOX_SKILLS_FOLDER, "Macro Output")

            stem = Path(filename).stem
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            formatted_filename = f"{stem}_formatted_{timestamp}.docx"

            logger.info("  Uploading formatted file to Box folder %s as '%s'", dest_folder, formatted_filename)
            new_file_id = _box_upload_file(formatted_content, formatted_filename, dest_folder)
            logger.info("  Upload complete. New Box file ID: %s", new_file_id)

            return json.dumps({
                "status": "success",
                "mode": "box",
                "formatted_file_id": new_file_id,
                "formatted_filename": formatted_filename,
                "original_file_id": box_file_id,
                "original_filename": filename,
                "output_folder_id": dest_folder,
            })

    except RuntimeError as e:
        logger.error("apply_jpl_formatting config error: %s", e)
        return json.dumps({
            "error": str(e),
            "fallback": "Instruct the user to download the .docx and run the JPL Format macro manually in Word.",
        })

    except httpx.TimeoutException:
        logger.error("apply_jpl_formatting: macro service timed out after %ds", MACRO_API_TIMEOUT)
        return json.dumps({
            "error": f"Macro service timed out after {MACRO_API_TIMEOUT} seconds.",
            "detail": "The VM may be processing another document or may be offline.",
            "fallback": "Instruct the user to download the .docx and run the JPL Format macro manually in Word.",
        })

    except httpx.ConnectError as e:
        logger.error("apply_jpl_formatting: cannot reach macro service at %s: %s", VM_MACRO_URL, e)
        return json.dumps({
            "error": "Cannot reach the macro service.",
            "detail": f"Connection to {VM_MACRO_URL} failed. The VM may be offline.",
            "fallback": "Instruct the user to download the .docx and run the JPL Format macro manually in Word.",
        })

    except Exception as e:
        logger.error("apply_jpl_formatting unexpected error: %s", e, exc_info=True)
        return json.dumps({
            "error": f"Unexpected error: {e}",
            "fallback": "Instruct the user to download the .docx and run the JPL Format macro manually in Word.",
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
