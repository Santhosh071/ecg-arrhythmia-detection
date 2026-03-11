# 🫀 ECG Arrhythmia Detection System

## 📌 What This Project Does

This system takes a raw ECG signal as input and outputs:

- Beat-level classification — identifies Normal, PVC, LBBB, RBBB, and 4 other arrhythmia types
- Anomaly detection — flags unusual beats even when the class is uncertain
- Risk scoring — assigns Low / Medium / High / Critical risk per session
- Clinical explanation — LLaMA3-70B explains findings in plain English
- Automated alerts — saves alert files and notifies when risk is elevated
- PDF reports — auto-generates session summary for clinician review
- Patient history — stores all sessions in SQLite, tracks trends over time

> ⚕️ **Clinical Disclaimer:** This system is a decision-support tool only.
> All outputs must be reviewed and confirmed by a qualified clinician before any action is taken.

---

## 📊 Model Performance

> Evaluated on the **MIT-BIH Arrhythmia Database** test split — 16,196 beats across 8 classes.

### CNN Classifier — 97.18% Accuracy

| Class | Arrhythmia Type | Precision | Recall | F1 Score | Samples |
|:---:|---|:---:|:---:|:---:|:---:|
| N | Normal Sinus Rhythm | 0.9969 | 0.9662 | 0.9813 | 11,255 |
| L | Left Bundle Branch Block | 0.9967 | 0.9950 | 0.9959 | 1,211 |
| R | Right Bundle Branch Block | 0.9963 | 0.9972 | 0.9968 | 1,089 |
| A | Atrial Premature Beat | 0.6568 | 0.9319 | 0.7706 | 382 |
| V | Premature Ventricular Contraction | 0.9531 | 0.9691 | 0.9610 | 1,069 |
| / | Paced Beat | 1.0000 | 1.0000 | **1.0000** | 1,054 |
| E | Ventricular Escape Beat | 0.9333 | 0.8750 | 0.9032 | 16 |
| F | Fusion Beat | 0.3931 | 0.9500 | 0.5561 | 120 |

### Autoencoder Anomaly Detection

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Transformer AE** | 98.36% | 0.9733 | 0.9729 | **0.9731** | **0.9872** |
| LSTM AE (baseline) | 68.95% | 0.3736 | 0.0263 | 0.0492 | 0.6195 |

### Combined System — Best for Clinical Use

| Metric | Value | Meaning |
|---|:---:|---|
| **Recall** | **99.57%** | Catches nearly all arrhythmias |
| **False Negative Rate** | **0.43%** | Only 21 missed out of 4,941 anomalies |
| False Positive Rate | 5.46% | Acceptable over-alerting |
| F1 Score | 0.9393 | Strong overall balance |

> In clinical systems, **recall is the most critical metric** — missing an arrhythmia is far more dangerous than a false alarm. The combined system achieves 99.57% recall on CPU-only hardware.

---

## 🏗 System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║               ECG Arrhythmia Detection — 6-Layer System              ║
╠══════════╦══════════╦══════════╦══════════╦══════════╦══════════════╣
║ LAYER 1  ║ LAYER 2  ║ LAYER 3  ║ LAYER 4  ║ LAYER 5  ║   LAYER 6   ║
║   DATA   ║  PREPROC ║  MODELS  ║  AGENT   ║DASHBOARD ║   STORAGE   ║
╠══════════╬══════════╬══════════╬══════════╬══════════╬══════════════╣
║ MIT-BIH  ║ Bandpass ║   CNN    ║LangChain ║Streamlit ║   SQLite    ║
║ Database ║ Filter   ║Classifier║  ReAct   ║ Live ECG ║  Sessions   ║
║ 48 ECGs  ║ 0.5-40Hz ║ 8 Class  ║  Agent   ║ Waveform ║  Alerts     ║
║ 360 Hz   ║          ║          ║          ║          ║  History    ║
║          ║ Baseline ║   +      ║   +      ║   +      ║             ║
║ 8 Classes║ Wander   ║Transform.║  Groq    ║ Alert    ║  PDF        ║
║ N L R A  ║ Correct  ║   AE     ║ LLaMA3   ║  Feed    ║  Reports    ║
║ V / E F  ║          ║ Anomaly  ║  70B     ║          ║             ║
║          ║ R-Peak   ║  Detect  ║ 8 Tools  ║   +      ║  outputs/  ║
║          ║ Detect   ║          ║          ║ Chat UI  ║  alerts/   ║
║          ║ (Pan-    ║   +      ║  Online: ║          ║             ║
║          ║ Tompkins)║  LSTM AE ║  Groq    ║   +      ║  Google    ║
║          ║          ║ Baseline ║ Offline: ║ Patient  ║  Drive     ║
║          ║ Segment  ║          ║  Phi-3   ║ History  ║  Backup    ║
║          ║ 187 samp ║   +      ║  Mini    ║ Charts   ║             ║
║          ║ Normalise║  VAE     ║          ║          ║             ║
║          ║ z-score  ║ Synth Aug║          ║ PDF DL   ║             ║
╚══════════╩══════════╩══════════╩══════════╩══════════╩══════════════╝
                              ▲
               ┌──────────────┴──────────────┐
               │    FastAPI REST Backend      │
               │    10 endpoints · CORS       │
               │    http://localhost:8000     │
               └─────────────────────────────┘
```

### Beat Decision Pipeline

```
Raw ECG Signal (360 Hz)
        │
        ▼
Preprocessing: Bandpass → Baseline Correction → R-Peak Detection → Segment → Normalise
        │
        ▼
Beat (187 samples × N beats)
        │
        ├──────────────────────────────────────────┐
        │                                          │
        ▼                                          ▼
CNN Classifier                          Transformer Autoencoder
(shape: batch, 1, 187)                  (shape: batch, 187)
        │                                          │
        ▼                                          ▼
Softmax → 8 class probs             Reconstruction Error (MSE)
class_id != 0 → flag                error > threshold → flag
        │                                          │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
             is_anomaly = CNN_flag OR AE_flag
                       │
                       ▼
             Risk Score Assessment
             ┌─────────────────────────────────────┐
             │ anomaly_rate < 5%  → 🟢 Low         │
             │ anomaly_rate < 10% → 🟡 Medium       │
             │ anomaly_rate < 30% → 🟠 High         │
             │ anomaly_rate ≥ 30% → 🔴 Critical     │
             │ dominant V or E    → Critical        │
             └─────────────────────────────────────┘
                       │
                       ▼
             Agent: assess_risk → send_alert → generate_report
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| PyTorch | 2.3.0 (CPU) | No GPU required |
| Groq API Key | Free tier | [Get here](https://console.groq.com) |
| RAM | 4GB+ | 12GB recommended |

### 1. Clone & Install

```bash
git clone https://github.com/Santhosh071/ecg-arrhythmia-detection.git
cd ecg-arrhythmia-detection

# Create virtual environment
python -m venv ecg
ecg\Scripts\activate          # Windows
# source ecg/bin/activate     # Linux / Mac

# Install all dependencies
pip install -r requirements.txt

# Install PyTorch CPU separately
pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Set Environment Variables

Copy `.env.template` to `.env` and fill in your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
PROJECT_ROOT=C:/ecg_arrhythmia
DATA_ROOT=D:/ecg_project
MODELS_PATH=D:/ecg_project/models
DB_PATH=D:/ecg_project/database/patient_history.db
```

### 3. Download Dataset

The MIT-BIH Arrhythmia Database is publicly available on PhysioNet:

👉 [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)

Or download directly via Python:

```bash
pip install wfdb
python src/layer1_data/download_mitbih.py
python src/layer2_preprocessing/preprocess_mitbih.py
```

### 4. Run the System

```bash
# Launch everything — API + Dashboard
python run.py

# Or run separately
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload    # Terminal 1
streamlit run src/layer5_dashboard/app.py                      # Terminal 2
```

---

## 🖥 Dashboard

The Streamlit dashboard has **4 tabs:**

| Tab | Features |
|---|---|
| 🫀 **Live Monitor** | ECG waveform (normal=blue, anomaly=red) · Risk badge · Alert feed · Beat inspector |
| 💬 **Clinician Chat** | Ask the AI agent anything · Quick action buttons · Conversation history |
| 📊 **Patient History** | Anomaly trend chart · Session table · Alert history |
| 📄 **Report** | Session summary · Generate PDF · Download button |

**How to use:**
1. Enter Patient ID in sidebar
2. Select Demo or upload a `.npy` file
3. Click **Run Analysis**
4. Explore results across all 4 tabs

---

## 🔌 API Endpoints

Interactive docs at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Models · Agent · DB · Groq status |
| `POST` | `/predict` | Batch beats → full session result |
| `POST` | `/predict/single` | One beat → class + confidence |
| `POST` | `/agent/query` | Natural language clinical Q&A |
| `POST` | `/report/generate` | Generate PDF session report |
| `POST` | `/signal/quality` | Raw signal quality analysis |
| `GET` | `/history/{patient_id}` | Past sessions |
| `GET` | `/history/{patient_id}/alerts` | Past alerts |
| `GET` | `/history/{patient_id}/trend` | Anomaly rate trend |

### Quick API Test

```python
import httpx, numpy as np

beats = np.load("data/mitbih/processed/test/beats.npy")[:10]
times = (np.arange(10) * (187 / 360)).tolist()

r = httpx.post("http://127.0.0.1:8000/predict", json={
    "patient_id" : "P001",
    "beats"      : beats.tolist(),
    "timestamps" : times,
    "fs"         : 360.0,
})
print(r.json()["session_risk"])   # Low / Medium / High / Critical
```

---

## 🤖 AI Agent & Tools

The **ECGAgent** uses LangChain ReAct with **Groq LLaMA3-70B** (free tier).
Automatically detects internet availability and switches to Phi-3 Mini offline.

### 8 Clinical Tools

| # | Tool | Type | Description |
|---|---|---|---|
| 1 | `monitor_trends` | Local | Anomaly frequency trend across sessions |
| 2 | `assess_risk` | Local | Low / Medium / High / Critical scoring |
| 3 | `explain_arrhythmia` | LLM | Plain-English clinical explanation |
| 4 | `send_alert` | Local | Saves alert to DB + `.txt` file |
| 5 | `generate_report` | Local | Auto PDF session summary |
| 6 | `answer_query` | LLM | Free-text clinical Q&A |
| 7 | `check_history` | DB | Past patient records from SQLite |
| 8 | `detect_sensor_issue` | Local | Signal quality flag |

### Memory System

- **Short-term:** `ConversationBufferWindowMemory(k=10)` — last 10 messages per session
- **Long-term:** SQLite database — all sessions, alerts, and agent logs persisted

---

## 🧪 Running Tests

```bash
# Run all tests
python tests/test_all.py

# Individual phase tests
python tests/test_inference.py      # Phase 5 — inference pipeline
python tests/test_agent.py          # Phase 6 — AI agent
python tests/test_integration.py    # Phase 9 — full pipeline
```

### Test Results

```
Phase 5 — Inference Pipeline    ✅  7/7 passed
Phase 6 — AI Agent              ✅  8/8 passed
Phase 9 — Integration           ✅  7/7 passed
─────────────────────────────────────────────
Total                           ✅  22/22 passed
```

---

## 📊 Running Evaluation

```bash
python evaluation/evaluate_models.py     # Compute all metrics
python evaluation/evaluation_report.py   # Generate PDF + CSV
```

---

## 📦 Dataset

| Dataset | Source | Records | Sample Rate |
|---|---|---|---|
| MIT-BIH Arrhythmia | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) | 48 recordings | 360 Hz |

| Split | Total Beats | Normal | Anomaly |
|---|---|---|---|
| Train | ~64,000 | ~45,000 | ~19,000 |
| Val | ~16,000 | ~11,000 | ~5,000 |
| Test | 16,196 | 11,255 | 4,941 |

---

## 🛠 Tech Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Deep Learning | PyTorch (CPU) | 2.3.0 |
| AI Agent | LangChain + LlamaIndex | 0.2.6 |
| LLM | Groq API — LLaMA3-70B | Free tier |
| Backend | FastAPI + Uvicorn | 0.111 |
| Dashboard | Streamlit + Plotly | 1.36 |
| Database | SQLite + SQLAlchemy | 2.0 |
| ECG Loading | wfdb | 4.1.2 |
| Signal Processing | scipy + neurokit2 | 1.13 |
| PDF Generation | ReportLab | 4.2 |
| Testing | pytest | 8.2 |

---
