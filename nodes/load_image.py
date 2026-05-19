import base64
import logging
from schemas import Agent_State

logger = logging.getLogger(__name__)


SUPPORTED_TYPES: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",   
    b"\x89PNG":      "image/png",    
    b"RIFF":         "image/webp",   
    b"GIF8":         "image/gif",    
}


MIN_IMAGE_BYTES = 1024  


def _detect_media_type(image_bytes: bytes) -> str | None:
    """
    Detect the MIME type of image_bytes by inspecting magic bytes.

    Returns the MIME type string, or None if the format is unsupported.
    """
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


def load_image_node(state: Agent_State) -> Agent_State:
    """
    LangGraph node: Load Image
    ──────────────────────────
    Responsibilities:
    1. Verify that image_bytes exist and are large enough.
    2. Detect and validate the image MIME type.
    3. Encode the image to base64 for downstream vision model use.
    4. Populate state fields: image_base64, image_media_type.
    5. Set state.error and state.status on failure.

    Args:
        state: Current Agent_State (must have image_bytes set)

    Returns:
        Updated Agent_State
    """
    logger.info(f"[load_image] Processing file: {state.filename}")

    
    if not state.image_bytes:
        msg = "No image bytes provided."
        logger.error(f"[load_image] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    
    if len(state.image_bytes) < MIN_IMAGE_BYTES:
        msg = f"Image is too small ({len(state.image_bytes)} bytes). Minimum is {MIN_IMAGE_BYTES} bytes."
        logger.error(f"[load_image] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    
    media_type = _detect_media_type(state.image_bytes)
    if media_type is None:
        # Fallback: try to infer from filename extension
        ext = state.filename.rsplit(".", 1)[-1].lower()
        ext_map = {
            "jpg":  "image/jpeg",
            "jpeg": "image/jpeg",
            "png":  "image/png",
            "webp": "image/webp",
        }
        media_type = ext_map.get(ext)

    if media_type is None:
        msg = (
            f"Unsupported image format for file '{state.filename}'. "
            "Accepted formats: JPEG, PNG, WebP."
        )
        logger.error(f"[load_image] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    logger.info(f"[load_image] Detected media type: {media_type}, size: {len(state.image_bytes):,} bytes")

    
    try:
        image_base64 = base64.b64encode(state.image_bytes).decode("utf-8")
    except Exception as e:
        msg = f"Failed to encode image: {e}"
        logger.error(f"[load_image] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    logger.info(f"[load_image] Image loaded successfully. Base64 length: {len(image_base64):,}")

    return state.model_copy(update={
        "image_base64":   image_base64,
        "image_media_type": media_type,
        "status":         "processing",
        "error":          None,
    })
