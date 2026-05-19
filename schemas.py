from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class Ornament_Item(BaseModel):
    """Represents a single category of detected gold ornament."""
    item_type: str = Field(
        ...,
        description="Type of ornament: ring, bangle, chain, necklace, earring, bracelet, anklet, coin, or other"
    )
    quantity: int = Field(
        ...,
        ge=0,
        description="Number of items of this type detected in the image"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) for this detection"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any additional notes about this item type"
    )



class Detection_Result(BaseModel):
    """Complete result from the ornament detection step."""
    items: List[Ornament_Item] = Field(
        default_factory=list,
        description="List of all detected ornament types with quantities"
    )
    total_items: int = Field(
        default=0,
        description="Total count of all ornaments detected"
    )
    total_types: int = Field(
        default=0,
        description="Number of distinct ornament types detected"
    )
    analysis_notes: Optional[str] = Field(
        default=None,
        description="General notes or observations from the analysis"
    )
    raw_response: Optional[str] = Field(
        default=None,
        description="Raw response string from the LLM for debugging"
    )



class Agent_State(BaseModel):
    """State object passed between LangGraph nodes."""
    # Input fields
    filename: str = Field(
        ...,
        description="Original filename of the uploaded image"
    )
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw bytes of the uploaded image"
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image string for the vision model"
    )
    image_media_type: Optional[str] = Field(
        default=None,
        description="MIME type of the image (e.g., image/jpeg)"
    )

    # Processing fields
    detection_result: Optional[Detection_Result] = Field(
        default=None,
        description="Results from the detection node"
    )
    db_record_ids: List[int] = Field(
        default_factory=list,
        description="IDs of records saved to the database"
    )

    # Status tracking
    error: Optional[str] = Field(
        default=None,
        description="Error message if any node fails"
    )
    status: str = Field(
        default="pending",
        description="Current status: pending, processing, completed, error"
    )

    class Config:
        arbitrary_types_allowed = True
