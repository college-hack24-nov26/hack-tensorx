# ECGX — TensorX Clinical AI

An AI-powered ECG analysis platform built with FastAPI. Classifies ECG signals into NSR, AFib, or VTCC, provides LLM-generated clinical interpretations, and includes a RAG-powered medical chatbot.

---

## Features

- **ECG Classification** — SVC model trained on 2500-sample Lead II signals
- **LLM Analysis** — Groq-powered clinical interpretation per prediction
- **RAG Chatbot** — FAISS + LangChain chatbot grounded in uploaded medical PDFs
- **PDF Reports** — Downloadable clinical reports with ECG waveform visualization
- **History** — SQLite-backed prediction history with vitals
- **VTCC Webhook** — Automatic n8n webhook trigger on Ventricular Tachycardia detection

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
```

### 3. Add medical documents (for RAG chatbot)
Place PDF files into the `documents/` folder. The vector index is built automatically on first chatbot use and cached at `vectordb/faiss_index/`.

### 4. Run
```bash
python main.py
```
App runs at `http://localhost:8000`

---

## Project Structure

```
ECGX/
├── main.py                  # FastAPI app, page routes
├── database.py              # SQLite init, save/get/clear history
├── routes/
│   ├── predict.py           # POST /predict, POST /download-report
│   └── chatbot.py           # POST /chat, /chat/reset, /chat/rebuild
├── services/
│   ├── rag_chatbot.py       # RAG pipeline (FAISS + LangChain + Groq)
│   ├── groq_analysis.py     # LLM ECG interpretation via Groq API
│   ├── model_loader.py      # SVC model loader + predict
│   ├── preprocess.py        # ECG signal filtering pipeline
│   ├── input_validator.py   # CSV/JSON signal parsing & validation
│   ├── report_generator.py  # PDF report generation (fpdf2)
│   └── webhook_service.py   # n8n VTCC emergency webhook
├── model/
│   └── svc_model_V1.h5      # Trained SVC model (joblib)
├── documents/               # PDFs for RAG knowledge base
├── vectordb/                # FAISS index (auto-generated)
├── templates/               # Jinja2 HTML templates
├── static/                  # Static assets
└── .env                     # API keys and config
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
| POST | `/predict` | Upload ECG file, returns prediction + LLM analysis |
| POST | `/download-report` | Generate and download PDF report |
| POST | `/chat` | Send message to RAG chatbot |
| POST | `/chat/reset` | Clear chatbot conversation memory |
| POST | `/chat/rebuild` | Rebuild FAISS vector index from documents |
| POST | `/history/clear` | Clear all prediction history |

---

## ECG Prediction Pipeline

1. Upload CSV or JSON file containing 2500 ECG samples (Lead II, 250Hz)
2. Signal is validated and parsed by `input_validator`
3. `clean_ecg_signal` applies 3-stage filtering for UI display:
   - Median filter baseline wander removal
   - Butterworth bandpass (0.5–40Hz)
   - Savitzky-Golay smoothing
4. `preprocess_signal` reshapes raw signal to `(1, 2500)` for the SVC model
5. SVC predicts class; probabilities derived via softmax over decision scores
6. Groq LLM generates clinical interpretation and recommended steps
7. If prediction is **VTCC**, n8n webhook is triggered with patient + analysis data
8. Result saved to SQLite; filtered signal returned for waveform chart

**Supported input formats:**
- `.csv` — single row of 2500 values, or a `signal` column, or 2500 columns per row
- `.json` — `{"signal": [...]}`, `[...]`, or array of signal arrays

---

## RAG Chatbot

Built with LangChain, FAISS, and Groq (`llama-3.3-70b-versatile`).

- PDFs in `documents/` are loaded, chunked (800 tokens, 150 overlap), and embedded using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS index is persisted to `vectordb/faiss_index/` and reused on restart
- Each query retrieves top-5 relevant chunks and passes them as context to the LLM
- Maintains last 3 conversation exchanges for context continuity
- To rebuild the index after adding new PDFs: `POST /chat/rebuild`

---

## Models & Keys

| Key | Description |
|-----|-------------|
| `GROQ_API_KEY` | Groq API key for LLM calls |
| `WEBHOOK_URL` | n8n webhook URL for VTCC alerts |

- ECG model: `model/svc_model_V1.h5` (scikit-learn SVC, loaded via joblib)
- LLM for ECG analysis: `openai/gpt-oss-120b` via Groq
- LLM for chatbot: `llama-3.3-70b-versatile` via Groq
