import os
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from database import init_db, get_recent_detections, get_stats
from schemas import Agent_State, Detection_Result
from graph import ornament_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(" Gold Ornament Detector starting up...")
    if not os.getenv("GROQ_API_KEY"):
        logger.error("  GROQ_API_KEY is not set. Detection will fail.")
    else:
        logger.info(" GROQ_API_KEY loaded.")
    init_db()
    logger.info(" Database initialized.")
    yield
    logger.info(" Gold Ornament Detector shutting down.")


app = FastAPI(title="Gold Ornament Detector", version="1.0.0", lifespan=lifespan)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Starlette 1.0.0+ — use request= kwarg, not {"request": request} in context dict
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    logger.info(f" Received upload: {file.filename} ({file.content_type})")

    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail=f"Unsupported file type. Accepted: JPEG, PNG, WebP.")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum is 20 MB.")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    initial_state = Agent_State(
        filename=file.filename or "unknown.jpg",
        image_bytes=image_bytes,
    )

    try:
        logger.info(f" Running detection pipeline for: {file.filename}")
        result_dict: dict = ornament_graph.invoke(initial_state)
        final_state = Agent_State(**{
            k: v for k, v in result_dict.items()
            if k in Agent_State.model_fields
        })
    except Exception as e:
        logger.error(f"Pipeline failed: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": f"{type(e).__name__}: {e}"})

    if final_state.error and not final_state.detection_result:
        return JSONResponse(status_code=422, content={"detail": final_state.error})

    detection: Detection_Result = final_state.detection_result
    if not detection:
        return JSONResponse(status_code=500, content={"detail": "Detection produced no results."})

    response_data = {
        "filename":       file.filename,
        "status":         final_state.status,
        "total_items":    detection.total_items,
        "total_types":    detection.total_types,
        "analysis_notes": detection.analysis_notes,
        "items": [
            {
                "item_type":  item.item_type,
                "quantity":   item.quantity,
                "confidence": round(item.confidence, 3) if item.confidence else None,
                "notes":      item.notes,
            }
            for item in detection.items
        ],
        "db_record_ids": final_state.db_record_ids,
        "error":         final_state.error,
    }

    logger.info(f" Detection complete: {detection.total_items} items for '{file.filename}'")
    return JSONResponse(content=response_data)


@app.get("/history")
async def get_history(limit: int = 50):
    limit = min(limit, 200)
    try:
        records = get_recent_detections(limit=limit)
        stats   = get_stats()
        return JSONResponse(content={"records": records, "stats": stats, "count": len(records)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status":       "ok",
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "version":      "1.0.0",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
