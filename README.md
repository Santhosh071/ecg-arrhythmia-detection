<div align="center">

# 🫀 ECG Arrhythmia Detection System

### A Production-Grade 6-Layer AI Pipeline for Real-Time Cardiac Monitoring

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2.6-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

[![MIT-BIH](https://img.shields.io/badge/Dataset-MIT--BIH%20Arrhythmia-blue?style=flat-square)](https://physionet.org/content/mitdb/1.0.0/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3--70B-orange?style=flat-square)](https://console.groq.com)
[![CPU Inference](https://img.shields.io/badge/Inference-CPU%20Only-green?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-22%2F22%20Passed-brightgreen?style=flat-square)]()

<br/>

> **Detects 8 types of cardiac arrhythmias from ECG signals using a multi-model AI ensemble —  
> CNN Classifier (97.18% accuracy) + Transformer Autoencoder (ROC-AUC 0.9872) +  
> LLM-powered clinical agent + live monitoring dashboard.**

<br/>

[📊 Model Results](#-model-performance) &nbsp;·&nbsp;
[🏗 Architecture](#-system-architecture) &nbsp;·&nbsp;
[🚀 Quick Start](#-quick-start) &nbsp;·&nbsp;
[🔌 API Docs](#-api-endpoints) &nbsp;·&nbsp;
[🤖 AI Agent](#-ai-agent--tools) &nbsp;·&nbsp;
[📁 Project Structure](#-project-structure)

</div>

---

## 📌 What This Project Does

This system takes a raw ECG signal as input and outputs:

- ✅ **Beat-level classification** — identifies Normal, PVC, LBBB, RBBB, and 4 other arrhythmia types
- ✅ **Anomaly detection** — flags unusual beats even when the class is uncertain
- ✅ **Risk scoring** — assigns Low / Medium / High / Critical risk per session
- ✅ **Clinical explanation** — LLaMA3-70B explains findings in plain English
- ✅ **Automated alerts** — saves alert files and notifies when risk is elevated
- ✅ **PDF reports** — auto-generates session summary for clinician review
- ✅ **Patient history** — stores all sessions in SQLite, tracks trends over time

> ⚕️ **Clinical Disclaimer:** This system is a decision-support tool only.  
> All outputs must be reviewed and confirmed by a qualified clinician before any action is taken.

---

## 📊 Model Performance

> Evaluated on the **MIT-BIH Arrhythmia Database** test split — 16,196 beats across 8 classes.

### 🔵 CNN Classifier — 97.18% Accuracy

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

### 🟠 Autoencoder Anomaly Detection

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Transformer AE** | 98.36% | 0.9733 | 0.9729 | **0.9731** | **0.9872** |
| LSTM AE (baseline) | 68.95% | 0.3736 | 0.0263 | 0.0492 | 0.6195 |

### 🟢 Combined System — Best for Clinical Use

| Metric | Value | Meaning |
|---|:---:|---|
| **Recall** | **99.57%** | Catches nearly all arrhythmias |
| **False Negative Rate** | **0.43%** | Only 21 missed out of 4,941 anomalies |
| False Positive Rate | 5.46% | Acceptable over-alerting |
| F1 Score | 0.9393 | Strong overall balance |

> 💡 In clinical systems, **recall is the most critical metric** — missing an arrhythmia is far more dangerous than a false alarm. The combined system achieves 99.57% recall on CPU-only hardware.

### ⚡ Inference Speed (Intel i5-6300U — CPU Only)

| Beat Count | Total Time | Per Beat |
|:---:|:---:|:---:|
| 10 beats | 229 ms | 22.9 ms |
| 50 beats | 972 ms | 19.4 ms |
| 100 beats | 1,958 ms | 19.6 ms |
| 500 beats | 9,868 ms | 19.7 ms |

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
║ PTB-XL   ║ Wander   ║Transform.║  Groq    ║ Alert    ║  PDF        ║
║ Database ║ Correct  ║   AE     ║ LLaMA3   ║  Feed    ║  Reports    ║
║21,799 ECG║          ║ Anomaly  ║  70B     ║          ║             ║
║ 500 Hz   ║ R-Peak   ║  Detect  ║ 8 Tools  ║   +      ║  outputs/  ║
║          ║ Detect   ║          ║          ║ Chat UI  ║  alerts/   ║
║ 8 Classes║ (Pan-    ║   +      ║  Online: ║          ║             ║
║ N L R A  ║ Tompkins)║  LSTM AE ║  Groq    ║   +      ║  Google    ║
║ V / E F  ║          ║ Baseline ║ Offline: ║ Patient  ║  Drive     ║
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

### 2. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and set:

```env
GROQ_API_KEY=your_groq_api_key_here     # Required — get free at console.groq.com
GROQ_MODEL_NAME=llama-3.3-70b-versatile
PROJECT_ROOT=C:/ecg_arrhythmia
DATA_ROOT=D:/ecg_project
MODELS_PATH=D:/ecg_project/models
DB_PATH=D:/ecg_project/database/patient_history.db
REPORTS_PATH=C:/ecg_arrhythmia/outputs/reports
```

### 3. Download Dataset

```bash
pip install wfdb
python src/layer1_data/download_mitbih.py
python src/layer2_preprocessing/preprocess_mitbih.py
```

Or download preprocessed data (77MB) directly:  
👉 **[Download Preprocessed MIT-BIH Beats](https://drive.google.com/your-link-here)** — place at `data/mitbih/processed/`

### 4. Download Pretrained Models

| Model | Download | Size | Metric |
|---|---|---|---|
| CNN Classifier | [Google Drive](https://drive.google.com/your-link) | ~15MB | 97.18% acc |
| Transformer AE | [Google Drive](https://drive.google.com/your-link) | ~25MB | AUC 0.9872 |
| LSTM AE | [Google Drive](https://drive.google.com/your-link) | ~10MB | Baseline |

Place all `.pth` files in `D:/ecg_project/models/` (or your configured `MODELS_PATH`).

### 5. Run the System

```bash
# Option A — Launch everything (API + Dashboard) with one command
python run.py

# Option B — Run separately
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload    # Terminal 1
streamlit run src/layer5_dashboard/app.py                      # Terminal 2

# Option C — API only
python run.py --api-only

# Option D — Dashboard only
python run.py --dashboard-only
```

### 6. Open in Browser

| Service | URL | Description |
|---|---|---|
| 🖥 Dashboard | http://localhost:8501 | Live ECG monitor + chat |
| 📡 API | http://localhost:8000 | REST backend |
| 📚 Swagger UI | http://localhost:8000/docs | Interactive API docs |
| ❤️ Health Check | http://localhost:8000/health | System status |

---

## 🖥 Dashboard

The Streamlit dashboard has **4 tabs:**

| Tab | Features |
|---|---|
| 🫀 **Live Monitor** | ECG waveform (normal=blue, anomaly=red) · Risk badge · Alert feed · Beat inspector · Class distribution |
| 💬 **Clinician Chat** | Ask the AI agent anything · Quick action buttons · Conversation history |
| 📊 **Patient History** | Anomaly trend chart · Session table · Alert history |
| 📄 **Report** | Session summary · Generate PDF · Download button |

**How to use:**
1. Enter Patient ID in sidebar
2. Select **Demo** or **Upload `.npy` file**
3. Click **▶️ Run Analysis**
4. Explore results across all 4 tabs

---

## 🔌 API Endpoints

Interactive docs at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info + links |
| `GET` | `/health` | Models · Agent · DB · Groq status |
| `POST` | `/predict` | Batch of beats → full session result |
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

# Load sample beats
beats = np.load("data/mitbih/processed/test/beats.npy")[:10]
times = (np.arange(10) * (187 / 360)).tolist()

# Run prediction
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
It automatically detects internet availability and switches providers.

### 8 Clinical Tools

| # | Tool | Type | Description |
|---|---|---|---|
| 1 | `monitor_trends` | Local | Anomaly frequency trend across sessions |
| 2 | `assess_risk` | Local | Low / Medium / High / Critical scoring |
| 3 | `explain_arrhythmia` | 🤖 LLM | Plain-English clinical explanation |
| 4 | `send_alert` | Local | Saves alert to DB + `.txt` file |
| 5 | `generate_report` | Local | Auto PDF session summary (ReportLab) |
| 6 | `answer_query` | 🤖 LLM | Free-text clinical Q&A |
| 7 | `check_history` | DB | Past patient records from SQLite |
| 8 | `detect_sensor_issue` | Local | Signal quality flag |

### Memory System

- **Short-term:** `ConversationBufferWindowMemory(k=10)` — last 10 messages per session
- **Long-term:** SQLite database — all sessions, alerts, agent logs persisted

### Example Chat

```
Clinician: "What is the clinical significance of PVC in this patient?"

Agent: "Premature Ventricular Contractions (PVCs) detected at 9.7% rate 
indicate ectopic beats originating in the ventricles. At this frequency, 
they may cause palpitations and reduced cardiac output. Recommend 24-hour 
Holter monitoring and evaluation for underlying structural heart disease..."
```

---

## 🧪 Running Tests

```bash
# Run all test suites (recommended before demo)
python tests/test_all.py

# Individual phase tests
python tests/test_inference.py      # Phase 5 — inference pipeline
python tests/test_agent.py          # Phase 6 — AI agent + tools
python tests/test_integration.py    # Phase 9 — full pipeline

# With API running (full end-to-end)
python tests/test_all.py --with-api
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
# Step 1 — Compute all metrics (takes 2-5 min on CPU)
python evaluation/evaluate_models.py

# Step 2 — Generate PDF report + CSV summary
python evaluation/evaluation_report.py
```

Output files:
```
evaluation/
    eval_results.json         ← all metrics in JSON
    evaluation_report.pdf     ← full PDF report
    evaluation_summary.csv    ← tables for capstone report
```

---

## 📁 Project Structure

```
ecg-arrhythmia-detection/
│
├── 📂 src/
│   ├── 📂 layer1_data/
│   │   ├── download_mitbih.py        # Download MIT-BIH via wfdb
│   │   └── download_ptbxl.py         # Download PTB-XL
│   │
│   ├── 📂 layer2_preprocessing/
│   │   ├── preprocess_mitbih.py      # Full preprocessing pipeline
│   │   └── signal_utils.py           # Filter, R-peak, segmentation
│   │
│   ├── 📂 layer3_models/
│   │   ├── cnn.py                    # 1D CNN Classifier (8 classes)
│   │   ├── transformer_ae.py         # Transformer Autoencoder
│   │   ├── lstm_ae.py                # LSTM Autoencoder (baseline)
│   │   ├── vae.py                    # VAE for rare class augmentation
│   │   └── 📂 inference/
│   │       ├── model_loader.py       # Load all 3 models at startup
│   │       ├── predict.py            # predict_beat / predict_beats
│   │       ├── preprocess.py         # Inference-time preprocessing
│   │       └── result_schema.py      # BeatResult / BatchResult
│   │
│   ├── 📂 layer4_agent/
│   │   ├── agent.py                  # ECGAgent (LangChain ReAct)
│   │   ├── agent_tools.py            # 8 clinical tools
│   │   ├── llm_config.py             # Groq config + internet check
│   │   └── memory.py                 # ECGDatabase (SQLite)
│   │
│   └── 📂 layer5_dashboard/
│       ├── app.py                    # Main Streamlit entry point
│       ├── ecg_plot.py               # Plotly ECG visualisation
│       ├── alert_panel.py            # Colour-coded alert feed
│       ├── chat_interface.py         # Clinician chat UI
│       └── history_charts.py         # Patient trend charts
│
├── 📂 api/
│   ├── main.py                       # FastAPI app (all 10 routes)
│   ├── schemas.py                    # Pydantic request/response models
│   └── dependencies.py               # Shared resource loader
│
├── 📂 notebooks/
│   ├── train_cnn.ipynb               # CNN training (Kaggle/Colab)
│   ├── train_transformer_ae.ipynb    # Transformer AE training
│   ├── train_lstm_ae.ipynb           # LSTM AE training
│   └── train_vae.ipynb               # VAE training
│
├── 📂 evaluation/
│   ├── evaluate_models.py            # Full metrics computation
│   └── evaluation_report.py         # PDF + CSV report generator
│
├── 📂 tests/
│   ├── test_all.py                   # Master test runner
│   ├── test_inference.py             # Phase 5 tests (7/7)
│   ├── test_agent.py                 # Phase 6 tests (8/8)
│   └── test_integration.py          # Phase 9 tests (7/7)
│
├── 📂 outputs/
│   ├── 📂 reports/                   # Auto-generated PDF reports
│   └── 📂 alerts/                    # Alert .txt files per patient
│
├── run.py                            # Single launcher (API + Dashboard)
├── requirements.txt                  # All dependencies
├── .env.example                      # Environment variable template
├── .gitignore                        # Excludes .env, .pth, .npy, .db
└── README.md
```

---

## 📦 Dataset & Models

### Dataset

| Dataset | Source | Records | Hz | Classes |
|---|---|---|---|---|
| MIT-BIH Arrhythmia | [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) | 48 ECGs | 360 Hz | 8 |
| PTB-XL | [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) | 21,799 ECGs | 500 Hz | 71 |

### Preprocessed Data (Ready to Use)

> Beat length: **187 samples** · Normalised: zero mean, unit variance

| Split | Beats | Normal | Anomaly |
|---|---|---|---|
| Train | ~64,000 | ~45,000 | ~19,000 |
| Val | ~16,000 | ~11,000 | ~5,000 |
| Test | 16,196 | 11,255 | 4,941 |

👉 **[Download Preprocessed Data (77MB)](https://drive.google.com/your-link)**

---

## ⚙️ Full Environment Variables

```env
# LLM (Required)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile

# Paths
PROJECT_ROOT=C:/ecg_arrhythmia
DATA_ROOT=D:/ecg_project
MODELS_PATH=D:/ecg_project/models
DB_PATH=D:/ecg_project/database/patient_history.db
REPORTS_PATH=C:/ecg_arrhythmia/outputs/reports
LOGS_PATH=D:/ecg_project/logs

# Signal processing constants (do not change)
BEAT_SIZE=187
MIT_BIH_SAMPLE_RATE=360
PTB_XL_SAMPLE_RATE=500
ANOMALY_STD_MULTIPLIER=2.0

# Server
API_HOST=127.0.0.1
API_PORT=8000
STREAMLIT_PORT=8501

# Risk thresholds
RISK_MEDIUM_THRESHOLD=0.10
RISK_HIGH_THRESHOLD=0.20

# Safety
ANONYMIZE_PATIENT_IDS=True
LOG_LEVEL=INFO
```

---

## 🛠 Full Tech Stack

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

## 🗺 Development Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project setup + folder structure | ✅ Done |
| Phase 2 | Data download + preprocessing pipeline | ✅ Done |
| Phase 3 | Model architectures (CNN, Transformer AE, LSTM AE, VAE) | ✅ Done |
| Phase 4 | Training notebooks (Kaggle / Colab) | ✅ Done |
| Phase 5 | Inference pipeline — 7/7 tests passed | ✅ Done |
| Phase 6 | AI Agent (tools + memory + LLM) — 8/8 tests passed | ✅ Done |
| Phase 7 | Streamlit dashboard | ✅ Done |
| Phase 8 | FastAPI backend | ✅ Done |
| Phase 9 | End-to-end integration — 7/7 tests passed | ✅ Done |
| Phase 10 | Evaluation + testing | ✅ Done |
| Phase 11 | Report writing + demo | 🔄 In Progress |

---

## 📄 License

This project is developed as an **academic capstone project** at  
**Vellore Institute of Technology (VIT), Chennai**  
Integrated M.Tech — Software Engineering

For academic and research purposes only. Not for clinical deployment.

---

## 🙋 Author

<div align="center">

**Santhoshkumar M**  
Integrated M.Tech Software Engineering — VIT Chennai  

[![GitHub](https://img.shields.io/badge/GitHub-Santhosh071-181717?style=for-the-badge&logo=github)](https://github.com/Santhosh071)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-santhoshkumar--m-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/santhoshkumar-m-b99360228)
[![Email](https://img.shields.io/badge/Email-santhoshkumarmurugan2004%40gmail.com-EA4335?style=for-the-badge&logo=gmail)](mailto:santhoshkumarmurugan2004@gmail.com)

</div>

---

<div align="center">

⭐ **Star this repo if you found it useful!**

*Built with ❤️ for cardiac health monitoring*

</div>
"# ecg-arrhythmia-detection" 
