# Gold Ornament Detector

An AI-powered FastAPI application that detects and counts gold ornaments in images using **Groq + LLaMA Vision** and a **LangGraph** pipeline.

## Features

-  **Expert AI Analysis** — Detailed gemologist-level prompt with spatial scanning and precise counting rules
-  **Beautiful UI** — Luxury dark-gold aesthetic with drag-and-drop upload, live preview, and animated results
-  **LangGraph Pipeline** — `load_image → detection → store_to_db → END`
-  **SQLite Persistence** — All detections saved with filename, type, quantity, confidence, and timestamp
-  **History View** — Last 50 detections with aggregate statistics
-  **Health Check** — API key and service status monitoring

## Supported Ornament Types

| Type | Description |
|------|-------------|
| `ring` | Finger rings — plain bands, stone-set, cluster, stacked sets |
| `bangle` | Rigid circular wrist ornaments |
| `chain` | Pure chain pieces without pendant |
| `necklace` | Chains with pendants, chokers, temple jewelry |
| `earring` | All ear ornaments — studs, hoops, jhumkas |
| `bracelet` | Flexible wrist ornaments — link, charm, tennis |
| `anklet` | Foot/ankle ornaments — payal, ankle chains |
| `coin` | Gold coins, medallions, sovereigns |
| `other` | Any other gold ornament |

## Project Structure

```
gold_ornament_detector/
├── main.py              # FastAPI app (endpoints, lifespan)
├── database.py          # SQLite setup and helpers
├── schemas.py           # Pydantic models (Ornament_Item, Detection_Result, Agent_State)
├── graph.py             # LangGraph workflow definition
├── nodes/
│   ├── __init__.py
│   ├── load_image.py    # Validate + base64-encode image
│   ├── detect_items.py  # Groq vision model detection
│   └── save_to_db.py    # Persist results to SQLite
├── templates/
│   └── index.html       # Jinja2 template (Tailwind + custom CSS)
├── static/              # Static assets (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Install uv (if you haven't already)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
# or: winget install astral-sh.uv                  # Windows
```

### 2. Clone / copy the project
```bash
cd gold_ornament_detector
```

### 3. Install dependencies
`uv` creates the virtualenv and installs everything from `pyproject.toml` in one step:
```bash
uv sync
```

To also install dev tools (pytest, ruff):
```bash
uv sync --group dev
```

### 4. Configure your API key
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com/keys).

### 5. Run the server
```bash
uv run uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Other useful uv commands
```bash
uv add <package>           # Add a new dependency
uv add --group dev <pkg>   # Add a dev-only dependency
uv remove <package>        # Remove a dependency
uv run pytest              # Run tests inside the venv
uv run ruff check .        # Lint the project
uv lock                    # Regenerate uv.lock without installing
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main UI |
| `POST` | `/analyze` | Upload and analyze an image |
| `GET` | `/history?limit=50` | Recent detection history |
| `GET` | `/health` | Health check |

### POST /analyze

**Request:** `multipart/form-data` with `file` field (JPEG, PNG, or WebP, max 20 MB)

**Response:**
```json
{
  "filename": "ornaments.jpg",
  "status": "completed",
  "total_items": 12,
  "total_types": 3,
  "analysis_notes": "Clear image with good lighting...",
  "items": [
    { "item_type": "ring", "quantity": 5, "confidence": 0.97, "notes": "Stack of 5 rings" },
    { "item_type": "bangle", "quantity": 4, "confidence": 0.95, "notes": null },
    { "item_type": "earring", "quantity": 3, "confidence": 0.92, "notes": "1.5 pairs visible" }
  ],
  "db_record_ids": [1, 2, 3],
  "error": null
}
```

## Counting Rules (AI Prompt)

The expert gemologist prompt enforces:
- **Stacked items**: Each ring/bangle counted individually
- **Earrings**: Each earring = 1 unit (pair = 2)
- **Coins**: Row-by-row scan, left to right
- **Necklace vs Chain**: Any pendant → necklace; pure chain → chain
- **Partial visibility**: >50% visible → counted; <50% → noted only
- **No reflections**: Mirror images not counted

## Tech Stack

- **FastAPI** — Web framework
- **LangGraph** — Stateful AI pipeline
- **Groq** — Ultra-fast LLM inference
- **LLaMA Scout 17B** — Vision model for ornament detection
- **SQLite** — Zero-config persistence
- **Pydantic v2** — Data validation
- **Tailwind CSS** — Utility-first styling
- **Cormorant Garamond** — Luxury display typography
