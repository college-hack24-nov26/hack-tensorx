# ECGX — TensorX Clinical AI

An AI-powered ECG analysis platform built with FastAPI. Classifies ECG signals into NSR, AFib, or VTCC, provides LLM-generated clinical interpretations, includes a RAG-powered medical chatbot, and supports real-time live ECG monitoring via a WebSocket bridge to an external ECG simulator.

---

## Features

- **ECG Classification** — SVC model trained on 2500-sample Lead II signals (250Hz)
- **LLM Analysis** — Groq-powered clinical interpretation per prediction (`openai/gpt-oss-120b`)
- **RAG Chatbot** — FAISS + LangChain chatbot grounded in uploaded medical PDFs (`openai/gpt-oss-20b`)
- **Live ECG Monitor** — Real-time WebSocket bridge to an external ECG simulator (port 8002); predicts every 250 new samples with alert cooldown
- **PDF Reports** — Downloadable clinical reports with ECG waveform visualization (fpdf2 + matplotlib)
- **History** — SQLite-backed prediction history with vitals (HR, BP, SpO2)
- **Emergency Alerts** — Automatic n8n webhook + ESP32 buzzer trigger on VTCC or AFib detection
- **Batch Processing** — Multi-row CSV/JSON ECG batch prediction support

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
WEBHOOK_URL=http://localhost:5678/webhook-test/your-webhook-id
ESP32_IP=192.168.x.x   # optional: ESP32 buzzer IP
```

### 3. Add medical documents (for RAG chatbot)
Place PDF files into the `documents/` folder. The vector index is built automatically on first chatbot use and cached at `vectordb/faiss_index/`.

### 4. Run
```bash
python main.py
```
App runs at `http://localhost:8000` (also accessible on LAN via printed network IP).

---

## Project Structure

```
ECGX/
├── main.py                    # FastAPI app, page routes, startup
├── database.py                # SQLite init (with migration), save/get/clear history
├── routes/
│   ├── predict.py             # POST /predict, POST /download-report
│   ├── chatbot.py             # POST /chat, /chat/reset, /chat/rebuild
│   └── simulator_stream.py    # Live ECG bridge: WebSocket + REST endpoints
├── services/
│   ├── groq_analysis.py       # LLM ECG interpretation via Groq API
│   ├── rag_chatbot.py         # RAG pipeline (FAISS + LangChain + Groq)
│   ├── model_loader.py        # SVC model loader + predict (joblib)
│   ├── preprocess.py          # ECG signal filtering pipeline (4-stage)
│   ├── input_validator.py     # CSV/JSON signal parsing & validation
│   ├── report_generator.py    # PDF report generation (fpdf2 + matplotlib)
│   └── webhook_service.py     # n8n webhook + ESP32 buzzer (sync & async)
├── model/
│   └── svc_model_V1.h5        # Trained SVC model (joblib format)
├── documents/                 # PDFs for RAG knowledge base (7 cardiology docs)
├── vectordb/
│   └── faiss_index/           # FAISS index (auto-generated, persisted)
├── templates/                 # Jinja2 HTML templates
│   ├── base.html / navbar.html
│   ├── home.html / dashboard.html / about.html
│   ├── chatbot.html / history.html / live.html
│   └── index.html
├── static/                    # Static assets
├── ecg.db                     # SQLite database
└── .env                       # API keys and config
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/dashboard` | ECG upload & analysis dashboard |
| GET | `/chatbot` | RAG chatbot UI |
| GET | `/history` | Prediction history |
| GET | `/about` | About page |
| GET | `/live` | Live ECG monitor UI |
| POST | `/predict` | Upload ECG file, returns prediction + LLM analysis |
| POST | `/download-report` | Generate and download PDF report |
| POST | `/chat` | Send message to RAG chatbot |
| POST | `/chat/reset` | Clear chatbot conversation memory |
| POST | `/chat/rebuild` | Rebuild FAISS vector index from documents |
| POST | `/history/clear` | Clear all prediction history |
| POST | `/live/start` | Start WebSocket bridge to ECG simulator |
| POST | `/live/stop` | Stop the bridge |
| GET | `/live/status` | Bridge state (running, session, clients) |
| GET | `/live/active-session` | Proxy simulator's active session ID |
| WS | `/ws/live` | Dashboard WebSocket — subscribe for live data + predictions |

---

## ECG Prediction Pipeline

1. Upload CSV or JSON file containing 2500 ECG samples (Lead II, 250Hz)
2. Signal is validated and parsed by `input_validator`
3. `clean_ecg_signal` applies 4-stage filtering for UI display:
   - Two-pass median filter baseline wander removal (200ms + 600ms windows)
   - Butterworth bandpass (0.5–40Hz, order 4, zero-phase SOS)
   - Savitzky-Golay smoothing (window=21, polyorder=3)
   - Min-Max normalization to [-1, 1]
4. `preprocess_signal` reshapes raw (unfiltered) signal to `(1, 2500)` for the SVC model
5. SVC predicts class; probabilities derived via softmax over decision scores
6. Groq LLM generates clinical interpretation and recommended steps
7. If prediction is **VTCC** or **AFib**: ESP32 buzzer triggered + n8n webhook fired (async, non-blocking)
8. Result saved to SQLite; filtered signal returned for waveform chart

**Batch mode:** Multi-row CSV or JSON arrays are processed per-row; LLM analysis is skipped for batch.

**Supported input formats:**
- `.csv` — single row of 2500 values, or a `signal` column, or 2500 columns per row
- `.json` — `{"signal": [...]}`, `[...]`, or array of signal arrays

---

## Live ECG Monitor

A real-time bridge connects ECGX to an external ECG simulator running on port 8002.

- Dashboard connects via `WS /ws/live`
- `POST /live/start` validates the session ID against the simulator's REST API, then opens a WebSocket to `wss://<host>/ws/<session_id>` (with SSL bypass for self-signed certs, falls back to `ws://`)
- Incoming samples are buffered in a rolling 2500-sample window; prediction runs every 250 new samples (1 second at 250Hz)
- On VTCC or AFib detection: ESP32 buzzer + n8n webhook triggered with a 30-second cooldown per condition
- Bridge broadcasts `live`, `segment`, `alert`, and `bridge_status` events to all connected dashboard clients

---

## RAG Chatbot

Built with LangChain, FAISS, and Groq.

- PDFs in `documents/` are loaded, chunked (800 tokens, 150 overlap), and embedded using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS index is persisted to `vectordb/faiss_index/` and reused on restart
- Each query retrieves top-5 relevant chunks passed as context to the LLM
- Maintains last 3 conversation exchanges (6 messages) for context continuity
- To rebuild the index after adding new PDFs: `POST /chat/rebuild`

**Included knowledge base (7 documents):**
- AFib management in hospital emergency departments
- AARC clinical practice guideline on defibrillation during resuscitation
- Early defibrillation instructions
- Expedited stacked shocks vs. compressions-first for in-hospital cardiac arrest
- Focused nurse-defibrillation training for in-hospital cardiac arrest
- Nurse defibrillation in a coronary care unit
- General AFib reference

---

## Database Schema

SQLite table `history` (with auto-migration for new columns):

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment PK |
| timestamp | DATETIME | Prediction time |
| prediction | TEXT | NSR / AFib / VTCC |
| nsr | REAL | NSR probability |
| afib | REAL | AFib probability |
| vfib | REAL | VTCC probability |
| heart_rate | INTEGER | BPM (optional) |
| bp_sys | INTEGER | Systolic BP (optional) |
| bp_dia | INTEGER | Diastolic BP (optional) |
| spo2 | INTEGER | SpO2 % (optional) |

---

## Models & Keys

| Key | Description |
|-----|-------------|
| `GROQ_API_KEY` | Groq API key for all LLM calls |
| `WEBHOOK_URL` | n8n webhook URL for VTCC/AFib alerts |
| `ESP32_IP` | IP address of ESP32 buzzer device (optional) |

| Component | Model |
|-----------|-------|
| ECG classifier | `model/svc_model_V1.h5` — scikit-learn SVC, loaded via joblib |
| ECG LLM analysis | `openai/gpt-oss-120b` via Groq |
| RAG chatbot LLM | `openai/gpt-oss-20b` via Groq |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |

