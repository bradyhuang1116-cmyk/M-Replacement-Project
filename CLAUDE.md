# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NodexelOCR — Engineering drawing number batch replacement system. Takes engineering drawings (PDF/TIF/JPG/PNG) and replaces part numbers starting with `Y`/`X`/`B`/`H` with new revision numbers (prefix `H`). Two processing paths auto-switched: **vector PDF direct rewrite** (<0.5s, 100%) and **VLM OCR pixel rewrite** (8–10s, 90%+). Remaining <10% anomalies handled by the standalone `manual_editor` PySide6 desktop tool.

## Quick Start

```bash
# 1. Conda env (Python 3.10)
#    Activate: source activate mitsubishi  (or use full path below)

# 2. Start VLM inference container (auto-started by docker_manager.py on first job)
docker run -d --gpus all -p 8080:8080 nodexelocr:v1

# 3. Start backend API (port 8000)
.\start_v2.bat

# 4. Start frontend (port 3000)
cd dashboard && npm install && npm run dev
```

## Key Commands

### Backend
```bash
# Start API server
.\start_v2.bat                          # also available as .vbs (hidden window)
python start_v2.py                      # direct entry

# Direct file processing (no API)
python -c "from modules.batch_processor import process_single_file; print(process_single_file('path/to/file.tif', 'output_dir'))"
```

### Tests
```bash
# Queue + Worker smoke test (requires conda env, but NOT Docker/VLM/PaddleOCR — uses monkey-patch)
/path/to/miniconda3/envs/mitsubishi/python.exe test_queue_worker_smoke.py

# Manual editor tests
/path/to/miniconda3/envs/mitsubishi/python.exe -m pytest manual_editor/tests -v
```

### Manual Editor (standalone desktop tool)
```bash
# Launch
/path/to/miniconda3/envs/mitsubishi/python.exe manual_editor/main.py
```

### Frontend
```bash
cd dashboard
npm install
npm run dev        # dev server (port 3000)
npm run build      # production build
npm run lint       # ESLint
```

### Compilation (for delivery)
```bash
# Nuitka compile modules → .pyd (called from scripts/打包脚本)
python -m nuitka --module modules/batch_processor.py ...
# PyInstaller for manual_editor → ManualEditor.exe
```

## Architecture

### Processing Pipeline

```
Input file (PDF/TIF/JPG/PNG)
  │
  ├── PDF ──► is_vector_pdf()?
  │             ├── Yes ──► replace_text_in_pdf() ──► vector/ output
  │             └── No  ──► convert_pdf_to_tif(600dpi)
  │
  └── Image ──► load_file() → img_array
                  │
                  ├── _enhance_vertical_lines()
                  ├── detect_all_regions()
                  │     ├── material_code_column (red box: part number column)
                  │     ├── bottom_right_number  (green box: title block number)
                  │     ├── top_left_number      (orange box: corner number)
                  │     └── factory_note_area    (factory note area detection)
                  │
                  ├── detect_cyan_boxes() per red box row
                  ├── replace_in_all_regions()  (OCR + glyph replacement)
                  └── verify_output()           (post-replacement verification)
```

### Output File Naming

```
Vector PDF:  output/vector/{original_name}.pdf
OCR image:   output/ocr/{original_name}.tif
Debug:       output/{original_name}_debug_regions.jpg
Y-boxes CSV: output/y_boxes.csv
```

First-letter-based suppress rules (lines 119–131 of `batch_processor.py`):
- **Y** prefix → all regions output
- **B** prefix → suppress red box (material_code_column)
- **P/G/O/J** prefix → suppress green + orange (bottom_right_number, top_left_number)
- Other → all three color boxes suppressed (only factory note passes)

### API Routes (all under `/api/v1`)

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/drawings/process` | POST | Upload + process single file synchronously |
| `/jobs/sse` | GET | SSE streaming job status |
| `/jobs/start` | POST | Start batch from folder |
| `/jobs/stop` | POST | Cancel running job |
| `/gpu` | GET | GPU metrics via nvidia-smi |
| `/folders/browse` | POST | Browse directory tree |
| `/folders/scan` | GET | List supported files in a folder |
| `/system/stop-all` | POST | Stop job + VLM container |
| `/system/shutdown` | POST | Full system shutdown |
| `/system/restart` | POST | Restart backend process |

Frontend calls the API via `dashboard/src/lib/api.ts` which provides `apiFetch()` and `apiSSE()` (EventSource wrapper). SSE endpoint (`/jobs/sse`) streams `JobState` typed as `dashboard/src/types.ts`.

### Directory Structure

```
.
├── config.py                     # Global config (env-overridable via .env/dotenv)
├── start_v2.py / .bat / .vbs     # Backend launcher (.vbs runs hidden)
├── requirements_local.txt        # Host Python deps (paddlepaddle-gpu CUDA 12.9!)
│
├── api/                          # FastAPI backend
│   ├── main.py                   # App factory + CORS + route registration
│   ├── routes/                   # health, drawings, jobs, gpu, folders, system
│   ├── services/drawing_service  # DrawingService: upload → process → return
│   └── models/response.py        # Pydantic response models
│
├── modules/                      # Business logic (compiled → .pyd for delivery)
│   ├── batch_processor.py        # process_single_file() — main pipeline entry
│   ├── pdf_vector_handler.py     # Vector PDF fast path
│   ├── region_detector.py        # Region detection algorithm [DO NOT MODIFY]
│   ├── text_replacer.py          # OCR + glyph replacement [DO NOT MODIFY]
│   ├── factory_note_pixel.py     # Factory note area detection [DO NOT MODIFY]
│   ├── vlm_ocr_engine.py         # VLM OCR via vLLM HTTP (PaddleOCR-VL-1.5)
│   ├── docker_manager.py         # NodexelOCR container lifecycle
│   ├── job_queue.py              # SQLite job queue
│   ├── worker.py                 # Single-threaded queue consumer
│   ├── process_log.py            # Process log (O/N markers for PLM)
│   ├── watch_folder.py           # Folder watcher (testing only, not production)
│   ├── file_ingestion.py         # File loading + PDF→TIF conversion
│   └── filename_parser.py        # Drawing number parser
│
├── docker/                       # Inference container
│   ├── Dockerfile.nodexel        # NodexelOCR image (model COPY'd in, zero mount)
│   └── entrypoint.sh
│
├── dashboard/                    # Next.js 16 frontend (Tailwind v4, TypeScript)
│   └── src/
│       ├── app/                  # Pages: / (home), /datasets
│       ├── components/           # FileTable, FolderBrowser, GPUWidget, ReplacingCard, Sidebar
│       ├── hooks/                # useGPUInfo, useJobStatus
│       ├── lib/api.ts            # API client + SSE helper
│       └── types.ts              # Shared TS types
│
├── manual_editor/                # PySide6 desktop fallback (→ ManualEditor.exe)
│   ├── main.py                   # Entry point
│   └── app/
│       ├── main_window.py        # 3-panel layout (file list / original / editor)
│       ├── editor_view.py        # Interactive editing canvas
│       ├── box_item.py           # 4-state draggable box (locked/unlocked/geo/text)
│       ├── csv_loader.py         # Read y_boxes.csv
│       ├── filename_map.py       # HXXX-R.tif ↔ XXX.tif mapping
│       └── render.py             # Render text onto PIL image
│
├── scripts/                      # Build / packaging scripts
│   ├── compile_core.sh           # Nuitka compilation
│   ├── 离线打包.sh               # Offline delivery packaging
│   ├── install_service.bat       # Windows service registration via nssm.exe
│   └── nssm.exe                  # Non-Sucking Service Manager binary
│
├── fonts/                        # CJK fonts for glyph replacement
│   └── dingliesongtypeface20241217-2.ttf  # Primary replacement font
│
├── vllm_config.yaml              # vLLM inference hyperparams (COPY'd into Docker image)
└── .env.example                  # All configurable env vars
```

### Container Lifecycle

The VLM inference container (`nodexelocr:v1`) is managed entirely by `docker_manager.py`:
1. `ensure_vlm_ready()` — starts Docker Desktop if needed, launches container, waits for health check
2. `start_vllm_container()` — `docker run -d --gpus all -p 8080:8080 nodexelocr:v1`
3. `stop_vllm_container()` — `docker stop` + `docker rm`
4. Container is self-contained (model + config COPY'd into image, zero host mounts)

## Critical Constraints

| Constraint | Reason |
|---|---|
| **paddlepaddle-gpu must be CUDA 12.9 build** (`pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/`) | Wrong CUDA version (cu126) causes silent OCR failures — no error, all results empty |
| **Do NOT modify** `region_detector.py`, `text_replacer.py`, `factory_note_pixel.py` | Core algorithms that are stable and delivered; any changes require full regression |
| **Do NOT modify** `manual_editor/` internals | Independent product line; changes break the standalone exe build |
| **Use full conda Python path** instead of `conda run` | `conda run` has encoding issues on Windows with Chinese characters |
| **GPU required** (VRAM ≥ 12 GB) | VLM inference (PaddleOCR-VL-1.5) hard requirement |
| **Y number pattern**: `[Y/X/B/H](?=[A-Z0-9]*\d)[A-Z0-9]{6,}` | First letter + at least 6 alphanumeric chars containing ≥1 digit (rejects pure-alpha like "YARIABLE") |
| **Model weights are inside the Docker image** | Not on host — container is zero-mount for delivery |

## Env Configuration

All configurable via env vars (see `config.py` and `.env.example`):
- `FONT_PATH` / `PDF_FONT_PATH` — Font paths
- `VLLM_BASE_URL` / `VLLM_MODEL_NAME` — VLM engine (default: `http://localhost:8080/v1`, `PaddleOCR-VL-1.5-0.9B`)
- `DOCKER_*` — Container name/port/image
- `API_HOST` / `API_PORT` / `API_CORS_ORIGINS` — Network
- `DATA_DIR` / `QUEUE_DB_PATH` / `PROCESS_LOG_DB_PATH` / `WORKER_OUTPUT_DIR` — Persistence
- `WATCH_*` — Folder watcher dirs/intervals
- `WORKER_POLL_INTERVAL` / `WORKER_MAX_RETRY` — Worker settings

## Config.py

Central config module (`config.py`) that:
- Loads `.env` via `python-dotenv` (optional import — no crash if missing)
- Provides `make_pattern()` to generate Y-number regex for arbitrary prefix sets
- Exports `FUZZY_DIGIT_MAP` for symbol→digit visual fallback (e.g., `/`→`1`)
- Exports `DEFAULT_REGIONS` / `FALLBACK_REGIONS` with keyword-anchored detection areas
- The `set_ocr_mode()` interface is a no-op stub for backward compatibility (VLM runs via HTTP now)

## PLM Integration

- PLM Oracle DB connected via `oracledb` (thin mode, no Oracle client required)
- Process log table exposes O/N fields per drawing (O = OCR used, N = no OCR needed)
- PLM side responsible for writing back docnumber/work_seq; our `process_single_file()` is the interface
- Watch folder (`WATCH_INBOX_DIR` et al.) is PLM's drop-in/pickup mechanism
