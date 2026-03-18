"""
JPL Template System — MCP Server (v7)

Connects Claude to the JPL template library (Pinecone), the JPL
formatting macro service (Azure VM), and the template ingestion
pipeline (Azure VM).

Deployed on Railway. Claude Teams connects via streamable HTTP transport.

Tools:
  jpl_search_templates      — Semantic search with deduplication, metadata
                              hydration, score filtering, and enriched
                              evaluation metadata
  jpl_get_template          — Retrieve full metadata for a specific template
  apply_jpl_formatting      — Run the JPL formatting macro on a .docx file
                              via the Azure VM macro service
  run_template_pipeline     — Trigger the template ingestion pipeline on
                              the Azure VM (hopper mode)
  check_pipeline_status     — Check whether the pipeline is running, idle,
                              or completed, with log tail for diagnostics

Changes from v6:
  - Search quality controls: score floor (0.50 minimum cosine similarity)
    filters out noise before results reach Claude. Count cap (20 max
    deduplicated results) prevents overwhelming context.
  - Score-enriched results: each result now includes rank position and
    score_gap_to_next so Claude can read the score distribution.
    Response includes a score_summary header (top/bottom/spread/count)
    for quick calibration.
  - below_score_floor count in response tells Claude how many results
    were filtered, supporting gap detection ("12 results were below
    the relevance threshold").

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
VM_MACRO_URL = os.environ.get("VM_MACRO_URL", "")       # e.g. http://<VM-IP>:8443/format
VM_MACRO_API_KEY = os.environ.get("VM_MACRO_API_KEY", "")

# Pipeline trigger on the Azure VM (same server as macro, different endpoint)
VM_PIPELINE_URL = os.environ.get("VM_PIPELINE_URL", "")  # e.g. http://<VM-IP>:8443/run-pipeline

# Box JWT config — stored as a JSON string in the Railway env var
BOX_JWT_CONFIG = os.environ.get("BOX_JWT_CONFIG", "")

# Box folder for macro output
BOX_SKILLS_FOLDER = "365569381705"

# Timeout for the macro API call
MACRO_API_TIMEOUT = 120  # seconds

# Search quality controls
SEARCH_SCORE_FLOOR = 0.50   # Minimum cosine similarity — results below this are noise
SEARCH_MAX_RESULTS = 20     # Maximum deduplicated results returned to Claude

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
    "box_file_id",
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
# API clients — lazy initialization
# ---------------------------------------------------------------------------
# All clients are created on first use, not at import time. This ensures
# the server starts immediately even if Pinecone or OpenAI are temporarily
# unreachable. Previously, module-level Pinecone initialization blocked
# server startup during Pinecone outages (Railway logs showed repeated
# ConnectTimeoutError to api.pinecone.io, preventing the server from
# ever listening for requests).
# ---------------------------------------------------------------------------

_openai_client = None
_pinecone_index = None
_box_client = None


def _get_openai_client():
    """Create or return the cached OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        logger.info("OpenAI client initialized")
    return _openai_client


def _get_pinecone_index():
    """Create or return the cached Pinecone index client."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(PINECONE_INDEX)
        logger.info("Pinecone index '%s' connected", PINECONE_INDEX)
    return _pinecone_index


def _get_box_client():
    """Create or return the cached Box client."""
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

    folder = client.folders.create_folder(
        name=folder_name,
        parent=CreateFolderParent(id=parent_folder_id),
    )
    logger.info("Created Box folder '%s' (ID: %s)", folder_name, folder.id)
    return folder.id


# ---------------------------------------------------------------------------
# VM API helper — derive status URL from pipeline URL
# ---------------------------------------------------------------------------

def _get_pipeline_status_url() -> str:
    """Derive the /pipeline-status URL from VM_PIPELINE_URL.

    VM_PIPELINE_URL is e.g. http://128.203.187.19:8443/run-pipeline
    We replace /run-pipeline with /pipeline-status.
    """
    if not VM_PIPELINE_URL:
        return ""
    return VM_PIPELINE_URL.replace("/run-pipeline", "/pipeline-status")


# ---------------------------------------------------------------------------
# Search helper functions — deduplication, hydration, metadata extraction
# ---------------------------------------------------------------------------

def _deduplicate_results(matches) -> list[dict]:
    """Deduplicate Pinecone results by template_id."""
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
            if vector_type == "primary" and not entry["has_primary_metadata"]:
                entry["metadata"] = dict(meta)
                entry["has_primary_metadata"] = True

    return sorted(by_template.values(), key=lambda x: x["best_score"], reverse=True)


def _hydrate_metadata(deduplicated: list[dict]) -> list[dict]:
    """Fetch full primary-vector metadata for templates matched only via HyPE."""
    needs_hydration = [
        entry for entry in deduplicated
        if not entry["has_primary_metadata"]
    ]

    if not needs_hydration:
        return deduplicated

    primary_ids = [f"{entry['template_id']}__primary" for entry in needs_hydration]

    try:
        fetch_response = _get_pinecone_index().fetch(ids=primary_ids)
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

    return deduplicated


def _extract_evaluation_metadata(metadata: dict) -> dict:
    """Extract the evaluation subset of metadata fields."""
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
    top_k: int = 20,
    practice_area: str = "",
    document_type: str = "",
    include_full_metadata: bool = False,
) -> str:
    """Search the JPL template library by describing a legal scenario or
    document need in natural language.

    Returns matching templates ranked by relevance, DEDUPLICATED by template
    (each template appears once regardless of how many vectors matched), with
    ENRICHED metadata for intelligent evaluation.

    QUALITY CONTROLS:
      - Score floor: Results below 0.50 cosine similarity are filtered out
        (they are noise at that distance). The response reports how many
        were filtered so you can detect library gaps.
      - Count cap: Maximum 20 deduplicated results returned.
      - Each result includes rank, score, and score_gap_to_next so you can
        read the distribution shape (cluster vs. gradual falloff vs. one
        clear winner).

    Args:
        query: Natural language description of the legal scenario or
               document need.
        top_k: Number of raw Pinecone results before deduplication (1-50,
               default 20).
        practice_area: Optional filter (e.g. "Foreclosure", "Quiet Title").
        document_type: Optional filter (e.g. "Petition", "Motion", "Deed").
        include_full_metadata: If true, return ALL metadata fields.

    Returns:
        JSON object with score_summary, unique_template_count,
        below_score_floor count, and deduplicated ranked results.
    """
    try:
        embed_response = _get_openai_client().embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_vector = embed_response.data[0].embedding

        meta_filter = _build_filter(practice_area, document_type)
        query_kwargs: dict = {
            "vector": query_vector,
            "top_k": min(max(top_k, 1), 50),
            "include_metadata": True,
        }
        if meta_filter:
            query_kwargs["filter"] = meta_filter

        results = _get_pinecone_index().query(**query_kwargs)
        raw_matches = results.matches or []

        logger.info("Search: '%s' → %d raw matches", query[:80], len(raw_matches))

        if not raw_matches:
            return json.dumps({
                "unique_template_count": 0,
                "raw_match_count": 0,
                "results": [],
                "message": "No templates found matching your query.",
            })

        deduplicated = _deduplicate_results(raw_matches)
        deduplicated = _hydrate_metadata(deduplicated)

        # --- Score floor: discard results below minimum similarity ---
        pre_floor_count = len(deduplicated)
        deduplicated = [
            e for e in deduplicated
            if e["best_score"] >= SEARCH_SCORE_FLOOR
        ]
        filtered_out = pre_floor_count - len(deduplicated)

        # --- Count cap: limit returned results ---
        capped = len(deduplicated) > SEARCH_MAX_RESULTS
        deduplicated = deduplicated[:SEARCH_MAX_RESULTS]

        logger.info(
            "Search: %d raw → %d deduped → %d after floor (%.2f) → %d returned%s",
            len(raw_matches), pre_floor_count, len(deduplicated),
            SEARCH_SCORE_FLOOR, len(deduplicated),
            " (capped)" if capped else "",
        )

        if not deduplicated:
            return json.dumps({
                "unique_template_count": 0,
                "raw_match_count": len(raw_matches),
                "below_score_floor": filtered_out,
                "results": [],
                "message": (
                    f"No templates scored above the {SEARCH_SCORE_FLOOR} "
                    f"relevance threshold. {filtered_out} result(s) were "
                    f"below the floor."
                ),
            })

        # --- Build results with rank and score gaps ---
        formatted_results = []
        for rank, entry in enumerate(deduplicated, start=1):
            metadata = entry["metadata"]
            score = round(entry["best_score"], 4)

            if include_full_metadata:
                result_metadata = {
                    k: v for k, v in metadata.items()
                    if v is not None and v != "" and v != []
                }
            else:
                result_metadata = _extract_evaluation_metadata(metadata)

            # Gap to the next result (0.0 for the last one)
            if rank < len(deduplicated):
                next_score = deduplicated[rank]["best_score"]
                gap = round(score - next_score, 4)
            else:
                gap = 0.0

            result = {
                "rank": rank,
                "score": score,
                "score_gap_to_next": gap,
                "template_id": entry["template_id"],
                "match_types": entry["match_types"],
                **result_metadata,
            }
            formatted_results.append(result)

        # --- Score summary for quick calibration ---
        scores = [r["score"] for r in formatted_results]
        score_summary = {
            "top_score": scores[0],
            "bottom_score": scores[-1],
            "score_spread": round(scores[0] - scores[-1], 4),
            "count": len(scores),
        }

        return json.dumps({
            "unique_template_count": len(formatted_results),
            "raw_match_count": len(raw_matches),
            "below_score_floor": filtered_out,
            "score_summary": score_summary,
            "results": formatted_results,
        })

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
    """Retrieve the full metadata for a specific template by its ID.

    Accepts either the base template_id or the full Pinecone vector ID
    with __primary suffix.

    Args:
        template_id: The template ID from search results.

    Returns:
        JSON object with the template's full metadata.
    """
    try:
        ids_to_try = [template_id]
        if "__" not in template_id:
            ids_to_try.append(f"{template_id}__primary")

        for candidate_id in ids_to_try:
            result = _get_pinecone_index().fetch(ids=[candidate_id])

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

    TWO MODES — use one:

    Mode 1 (PREFERRED for chat sessions): Pass file_content_base64.
    Mode 2 (for pipeline/Box workflows): Pass box_file_id.

    Args:
        file_content_base64: Base64-encoded .docx file content.
        filename: Filename for the document (default: document.docx).
        box_file_id: Box file ID of the .docx to format.
        output_folder_id: (Box mode only) Box folder ID for the result.

    Returns:
        JSON with status and formatted file (base64 or Box file ID).
    """
    import base64
    from datetime import datetime as dt

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
        if file_content_base64:
            mode = "direct"
            try:
                file_content = base64.b64decode(file_content_base64)
            except Exception as e:
                return json.dumps({"error": f"Invalid base64 content: {e}"})
            if not filename.lower().endswith(".docx"):
                filename = filename + ".docx"
            logger.info("apply_jpl_formatting (direct mode): %s (%s bytes)",
                        filename, f"{len(file_content):,}")
        else:
            mode = "box"
            logger.info("apply_jpl_formatting (box mode): downloading Box file %s", box_file_id)
            file_content, filename = _box_download_file(box_file_id)
            if not filename.lower().endswith(".docx"):
                return json.dumps({
                    "error": f"File '{filename}' is not a .docx file.",
                    "detail": "The macro service only accepts .docx files.",
                })
            logger.info("  Downloaded: %s (%s bytes)", filename, f"{len(file_content):,}")

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

        if mode == "direct":
            formatted_base64 = base64.b64encode(formatted_content).decode("ascii")
            return json.dumps({
                "status": "success",
                "mode": "direct",
                "formatted_file_base64": formatted_base64,
                "filename": filename,
            })
        else:
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
# Tool: Run Template Pipeline
# ---------------------------------------------------------------------------

@mcp.tool(
    name="run_template_pipeline",
    annotations={
        "title": "Run Template Ingestion Pipeline",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def run_template_pipeline() -> str:
    """Trigger the JPL template ingestion pipeline on the Azure VM.

    This runs the pipeline in hopper mode — it processes ALL documents
    currently in the Box hopper folder (deposited by the intake skill
    or manually). The pipeline runs as a background process on the VM;
    this tool returns immediately with confirmation that it started.

    Each document goes through the full chain: deep analysis → metadata
    extraction → search optimization → templatization → formatting →
    macro → .dotx conversion → Box upload + Pinecone indexing.
    Typical time: 15-20 minutes per document.

    WHEN TO CALL:
      - After depositing documents in the hopper folder via the intake skill
      - After the user confirms they want to run the pipeline
      - Do NOT call if no documents have been deposited — the pipeline
        will find an empty hopper and exit immediately

    Returns:
        JSON with trigger status:
        - "triggered": Pipeline started successfully. Tell the user their
          template(s) should appear in the library within 15-20 minutes
          per document.
        - "already_running": A pipeline run is already in progress. Tell
          the user to wait for it to finish. The pipeline re-scans the
          hopper after each batch, so new documents will be picked up
          automatically.
        - "error": Something went wrong. Show the error and offer the
          manual fallback: run `py pipeline_v3.py --hopper` on the VM.
    """
    if not VM_PIPELINE_URL:
        return json.dumps({
            "error": "Pipeline trigger not configured.",
            "detail": "VM_PIPELINE_URL environment variable is not set on the MCP server.",
            "fallback": "Run the pipeline manually on the VM: py pipeline_v3.py --hopper",
        })

    try:
        headers = {}
        if VM_MACRO_API_KEY:
            headers["Authorization"] = f"Bearer {VM_MACRO_API_KEY}"

        logger.info("run_template_pipeline: triggering %s", VM_PIPELINE_URL)

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(VM_PIPELINE_URL, headers=headers)

        if response.status_code == 409:
            data = response.json()
            logger.info("run_template_pipeline: already running (%s)", data.get("started_at"))
            return json.dumps({
                "status": "already_running",
                "message": data.get("message", "Pipeline is already running."),
                "started_at": data.get("started_at"),
                "elapsed_seconds": data.get("elapsed_seconds"),
            })

        if response.status_code != 200:
            error_detail = response.text[:500] if response.text else "No response body"
            logger.error("run_template_pipeline: VM returned %d: %s", response.status_code, error_detail)
            return json.dumps({
                "error": f"VM returned HTTP {response.status_code}.",
                "detail": error_detail,
                "fallback": "Run the pipeline manually on the VM: py pipeline_v3.py --hopper",
            })

        data = response.json()
        logger.info("run_template_pipeline: triggered (PID %s)", data.get("pid"))
        return json.dumps({
            "status": "triggered",
            "message": data.get("message", "Pipeline triggered successfully."),
            "started_at": data.get("started_at"),
        })

    except httpx.ConnectError as e:
        logger.error("run_template_pipeline: cannot reach VM at %s: %s", VM_PIPELINE_URL, e)
        return json.dumps({
            "error": "Cannot reach the VM.",
            "detail": f"Connection to {VM_PIPELINE_URL} failed. The VM may be offline or the port may not be open.",
            "fallback": "Run the pipeline manually on the VM: py pipeline_v3.py --hopper",
        })

    except httpx.TimeoutException:
        logger.error("run_template_pipeline: request to VM timed out")
        return json.dumps({
            "error": "Request to VM timed out after 30 seconds.",
            "detail": "The VM may be offline or unresponsive.",
            "fallback": "Run the pipeline manually on the VM: py pipeline_v3.py --hopper",
        })

    except Exception as e:
        logger.error("run_template_pipeline unexpected error: %s", e, exc_info=True)
        return json.dumps({
            "error": f"Unexpected error: {e}",
            "fallback": "Run the pipeline manually on the VM: py pipeline_v3.py --hopper",
        })


# ---------------------------------------------------------------------------
# Tool: Check Pipeline Status
# ---------------------------------------------------------------------------

@mcp.tool(
    name="check_pipeline_status",
    annotations={
        "title": "Check Pipeline Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def check_pipeline_status() -> str:
    """Check the current status of the template ingestion pipeline.

    Reports whether the pipeline is running, idle, or completed, along
    with elapsed time and the last 30 lines of the pipeline log for
    diagnostics.

    Use this to monitor pipeline progress during batch processing or
    to check if a triggered run has completed.

    Returns:
        JSON with pipeline status:
        - "running": Pipeline is actively processing. Includes elapsed
          time and log tail showing current activity.
        - "completed": Pipeline finished. Includes exit code and log tail.
        - "idle": No pipeline run recorded since the API server started.
        - "error": Could not reach the VM to check status.
    """
    status_url = _get_pipeline_status_url()

    if not status_url:
        return json.dumps({
            "error": "Pipeline status not configured.",
            "detail": "VM_PIPELINE_URL environment variable is not set on the MCP server.",
            "fallback": "Check the pipeline log on the VM: type \"C:\\JPL Templates System Pipeline\\pipeline_latest.log\"",
        })

    try:
        headers = {}
        if VM_MACRO_API_KEY:
            headers["Authorization"] = f"Bearer {VM_MACRO_API_KEY}"

        logger.info("check_pipeline_status: querying %s", status_url)

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(status_url, headers=headers)

        if response.status_code != 200:
            error_detail = response.text[:500] if response.text else "No response body"
            logger.error("check_pipeline_status: VM returned %d: %s", response.status_code, error_detail)
            return json.dumps({
                "error": f"VM returned HTTP {response.status_code}.",
                "detail": error_detail,
            })

        data = response.json()
        logger.info("check_pipeline_status: %s", data.get("status"))
        return json.dumps(data)

    except httpx.ConnectError as e:
        logger.error("check_pipeline_status: cannot reach VM: %s", e)
        return json.dumps({
            "error": "Cannot reach the VM.",
            "detail": f"Connection failed. The VM may be offline.",
            "fallback": "Check the pipeline log on the VM: type \"C:\\JPL Templates System Pipeline\\pipeline_latest.log\"",
        })

    except httpx.TimeoutException:
        logger.error("check_pipeline_status: request timed out")
        return json.dumps({
            "error": "Request to VM timed out.",
            "detail": "The VM may be offline or unresponsive.",
            "fallback": "Check the pipeline log on the VM: type \"C:\\JPL Templates System Pipeline\\pipeline_latest.log\"",
        })

    except Exception as e:
        logger.error("check_pipeline_status unexpected error: %s", e, exc_info=True)
        return json.dumps({
            "error": f"Unexpected error: {e}",
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
    logger.info("Starting JPL Template MCP server v7 on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
