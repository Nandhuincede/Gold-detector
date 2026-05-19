import logging
from schemas import Agent_State
from database import insert_detection

logger = logging.getLogger(__name__)


def save_to_db_node(state: Agent_State) -> Agent_State:
    """
    LangGraph node: Save to Database
    ──────────────────────────────────
    Responsibilities:
    1. Verify that a detection_result is available and has items.
    2. Insert one database row per detected ornament type.
    3. Collect and store the generated row IDs in state.db_record_ids.
    4. Mark state.status as 'completed' on success.

    Args:
        state: Current Agent_State (must have detection_result populated)

    Returns:
        Updated Agent_State with db_record_ids and status='completed'.
    """
    logger.info(f"[save_to_db] Saving results for: {state.filename}")

    
    if state.error:
        logger.warning(f"[save_to_db] Skipping due to prior error: {state.error}")
        return state

    if not state.detection_result:
        msg = "No detection result to save. Run detect_items_node first."
        logger.error(f"[save_to_db] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    detection = state.detection_result

    
    if not detection.items:
        logger.info("[save_to_db] No ornaments detected; nothing to save.")
        return state.model_copy(update={
            "db_record_ids": [],
            "status":        "completed",
        })

    
    record_ids: list[int] = []
    errors: list[str] = []

    for ornament in detection.items:
        try:
            record_id = insert_detection(
                filename=state.filename,
                item_type=ornament.item_type,
                quantity=ornament.quantity,
                confidence=ornament.confidence,
                notes=ornament.notes,
            )
            record_ids.append(record_id)
            logger.debug(
                f"[save_to_db] Saved: {ornament.item_type} x{ornament.quantity} → id={record_id}"
            )
        except Exception as e:
            err_msg = f"Failed to save '{ornament.item_type}': {e}"
            logger.error(f"[save_to_db] {err_msg}")
            errors.append(err_msg)

    
    if errors:
        logger.warning(f"[save_to_db] {len(errors)} item(s) failed to save.")

    logger.info(
        f"[save_to_db] Saved {len(record_ids)}/{len(detection.items)} records "
        f"for file '{state.filename}'."
    )

    return state.model_copy(update={
        "db_record_ids": record_ids,
        "status":        "completed",
        "error":         "; ".join(errors) if errors else None,
    })
