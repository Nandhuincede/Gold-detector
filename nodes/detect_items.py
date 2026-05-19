import os
import json
import logging
import re
from groq import Groq
from schemas import Agent_State, Detection_Result, Ornament_Item

logger = logging.getLogger(__name__)


GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = """You are an expert gemologist and gold ornament specialist with 30+ years of experience
appraising and cataloging fine gold jewelry from South Asian, Middle Eastern, and Western traditions.
You have exceptional visual acuity for identifying and precisely counting gold ornaments in photographs.

YOUR TASK:
Analyze the provided image and perform a comprehensive gold ornament inventory with absolute precision.

SPATIAL SCANNING PROTOCOL:
1. Mentally divide the image into a 3×3 grid (top-left, top-center, top-right, middle-left, center,
   middle-right, bottom-left, bottom-center, bottom-right).
2. Systematically examine each cell of the grid before moving to the next.
3. After the grid scan, do a final full-image sweep to catch any items on borders.
4. Pay special attention to shadows, reflections, and partially obscured items.

ORNAMENT CATEGORIES (detect ONLY these types):
- ring       : Finger rings of any style — plain bands, stone-set, cluster, cocktail, stacked sets
- bangle     : Rigid circular wrist ornaments — plain, carved, filigree, stone-set
- chain      : Pure chain pieces without a pendant — link chains, rope chains, snake chains, figaro
- necklace   : Chains WITH pendants, chokers, collar necklaces, layered necklaces, temple jewelry
- earring    : ALL ear ornaments — studs, hoops, jhumkas, chandbalis, drops, ear cuffs
- bracelet   : Flexible wrist ornaments — link bracelets, charm bracelets, tennis bracelets, kadas
- anklet     : Foot/ankle ornaments — payal, ankle chains, anklet sets
- coin       : Gold coins, medallions, sovereign coins, gold discs
- other      : Any gold ornament that does not fit the above categories

CRITICAL COUNTING RULES:
1. STACKED / LAYERED ITEMS:
   - Count each individual piece, even if stacked or nested.
   - For stacked rings: count each ring separately (a stack of 5 rings = 5 rings).
   - For bangle sets: count each bangle individually (a chura/chooda set counts each bangle).

2. EARRINGS:
   - Count EACH earring as one unit.
   - A matching pair of earrings = 2 (quantity: 2).
   - If only one earring is visible, count it as 1.
   - Never assume a pair if only one is visible.

3. GOLD COINS:
   - Scan row by row, left to right, top to bottom.
   - Count each coin individually.
   - If coins overlap, estimate the total count carefully.

4. NECKLACE vs CHAIN:
   - If a chain has ANY pendant, locket, or center piece → necklace.
   - If it is purely chain with no pendant → chain.

5. PARTIAL VISIBILITY:
   - If more than 50% of an item is visible, count it.
   - If less than 50% visible, note it in analysis_notes but do NOT count it.

6. REFLECTIONS AND DUPLICATES:
   - Do NOT count mirror reflections as separate items.
   - Verify each item is a physical object, not its reflection.

7. ZERO QUANTITIES:
   - Only include categories that have at least 1 item (quantity > 0).
   - Do NOT include categories with quantity = 0.

OUTPUT FORMAT (strict JSON only, no markdown, no explanation outside JSON):
{
  "items": [
    {
      "item_type": "<category>",
      "quantity": <integer>,
      "confidence": <float between 0.0 and 1.0>,
      "notes": "<optional brief note about this item type>"
    }
  ],
  "total_items": <sum of all quantities>,
  "total_types": <number of distinct item_type entries>,
  "analysis_notes": "<overall notes about the image, lighting, visibility, or special observations>"
}

IMPORTANT: Respond ONLY with the JSON object. No preamble, no explanation, no markdown code fences."""


def _parse_detection_response(raw: str) -> Detection_Result:
    """
    Parse the raw LLM string response into a Detection_Result.
    Handles JSON extraction even if the model adds extra text.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object with regex as a last resort
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Could not parse JSON from response: {cleaned[:300]}")

    # Build Ornament_Item list
    items = []
    for item_data in data.get("items", []):
        item_type = str(item_data.get("item_type", "other")).lower().strip()
        quantity = int(item_data.get("quantity", 0))
        if quantity > 0:  # Only include non-zero quantities
            items.append(Ornament_Item(
                item_type=item_type,
                quantity=quantity,
                confidence=item_data.get("confidence"),
                notes=item_data.get("notes"),
            ))

    total_items = sum(i.quantity for i in items)
    total_types = len(items)

    return Detection_Result(
        items=items,
        total_items=total_items,
        total_types=total_types,
        analysis_notes=data.get("analysis_notes"),
        raw_response=raw,
    )


def detect_items_node(state: Agent_State) -> Agent_State:
    """
    LangGraph node: Detect Items
    ─────────────────────────────
    Responsibilities:
    1. Verify that image_base64 and image_media_type are available.
    2. Call the Groq API with the vision model and expert gemologist prompt.
    3. Parse the structured JSON response into a Detection_Result.
    4. Populate state.detection_result.
    5. Handle and report any API or parsing errors.

    Args:
        state: Current Agent_State (must have image_base64 and image_media_type set)

    Returns:
        Updated Agent_State with detection_result populated.
    """
    logger.info(f"[detect_items] Starting detection for: {state.filename}")

    
    if state.error:
        logger.warning(f"[detect_items] Skipping due to prior error: {state.error}")
        return state

    if not state.image_base64 or not state.image_media_type:
        msg = "Image not loaded (base64 or media_type missing). Run load_image_node first."
        logger.error(f"[detect_items] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        msg = "GROQ_API_KEY environment variable is not set."
        logger.error(f"[detect_items] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    client = Groq(api_key=api_key)

    
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{state.image_media_type};base64,{state.image_base64}"
                },
            },
            {
                "type": "text",
                "text": (
                    "Please perform a complete gold ornament inventory of this image. "
                    "Apply your full spatial scanning protocol and all counting rules. "
                    "Return ONLY the JSON response as specified."
                ),
            },
        ],
    }

    
    try:
        logger.info(f"[detect_items] Calling Groq model: {GROQ_MODEL}")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                user_message,
            ],
            temperature=0.1,   # Low temperature for precise, deterministic counting
            max_tokens=1500,
        )
        raw_response = response.choices[0].message.content
        logger.info(f"[detect_items] Raw response received ({len(raw_response)} chars)")
        logger.debug(f"[detect_items] Raw response: {raw_response[:500]}")
    except Exception as e:
        msg = f"Groq API call failed: {e}"
        logger.error(f"[detect_items] {msg}")
        return state.model_copy(update={"error": msg, "status": "error"})

    
    try:
        detection_result = _parse_detection_response(raw_response)
        logger.info(
            f"[detect_items] Detected {detection_result.total_items} items "
            f"across {detection_result.total_types} types."
        )
    except Exception as e:
        msg = f"Failed to parse detection response: {e}"
        logger.error(f"[detect_items] {msg}")
        # Still save the raw response for debugging
        return state.model_copy(update={
            "error": msg,
            "status": "error",
            "detection_result": Detection_Result(raw_response=raw_response),
        })

    return state.model_copy(update={
        "detection_result": detection_result,
        "status": "detected",
        "error": None,
    })
