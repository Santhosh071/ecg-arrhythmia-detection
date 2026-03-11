import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME   = os.getenv("GROQ_MODEL_NAME", "llama3-70b-8192")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "phi3")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "C:/ecg_arrhythmia"))
DATA_ROOT    = Path(os.getenv("DATA_ROOT",    "D:/ecg_project"))

MIT_BIH_PATH = Path(os.getenv("MIT_BIH_PATH", "D:/ecg_project/datasets/mitbih/raw"))
PTB_XL_PATH  = Path(os.getenv("PTB_XL_PATH",  "D:/ecg_project/datasets/ptbxl/raw"))

PROCESSED_DATA_PATH    = Path(os.getenv("PROCESSED_DATA_PATH",    "D:/ecg_project/datasets/mitbih/processed"))
MIT_BIH_PROCESSED_PATH = Path(os.getenv("MIT_BIH_PROCESSED_PATH", "D:/ecg_project/datasets/mitbih/processed"))
PTB_XL_PROCESSED_PATH  = Path(os.getenv("PTB_XL_PROCESSED_PATH",  "D:/ecg_project/datasets/ptbxl/processed"))

MODELS_PATH         = Path(os.getenv("MODELS_PATH",         "D:/ecg_project/models"))
TRANSFORMER_AE_PATH = Path(os.getenv("TRANSFORMER_AE_PATH", "D:/ecg_project/models/transformer_ae"))
CNN_CLASSIFIER_PATH = Path(os.getenv("CNN_CLASSIFIER_PATH", "D:/ecg_project/models/cnn_classifier"))
LSTM_AE_PATH        = Path(os.getenv("LSTM_AE_PATH",        "D:/ecg_project/models/lstm_ae"))
VAE_PATH            = Path(os.getenv("VAE_PATH",            "D:/ecg_project/models/vae"))
CHECKPOINTS_PATH    = Path(os.getenv("CHECKPOINTS_PATH",    "D:/ecg_project/models/checkpoints"))

REPORTS_PATH     = Path(os.getenv("REPORTS_PATH",     "D:/ecg_project/reports"))
LOGS_PATH        = Path(os.getenv("LOGS_PATH",        "D:/ecg_project/logs"))
DB_PATH          = Path(os.getenv("DB_PATH",          "D:/ecg_project/database/patient_history.db"))
LLM_WEIGHTS_PATH = Path(os.getenv("LLM_WEIGHTS_PATH", "D:/ecg_project/llm_weights/phi3_mini"))

BEAT_SIZE              = int(os.getenv("BEAT_SIZE",              187))
MIT_BIH_SAMPLE_RATE    = int(os.getenv("MIT_BIH_SAMPLE_RATE",    360))
PTB_XL_SAMPLE_RATE     = int(os.getenv("PTB_XL_SAMPLE_RATE",     500))
ANOMALY_STD_MULTIPLIER = float(os.getenv("ANOMALY_STD_MULTIPLIER", 2.0))
CLASS_NAMES            = os.getenv("CLASS_NAMES", "N,L,R,A,V,/,E,F").split(",")

API_HOST       = os.getenv("API_HOST",       "127.0.0.1")
API_PORT       = int(os.getenv("API_PORT",       8000))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))

RISK_LOW_THRESHOLD    = float(os.getenv("RISK_LOW_THRESHOLD",    0.05))
RISK_MEDIUM_THRESHOLD = float(os.getenv("RISK_MEDIUM_THRESHOLD", 0.10))
RISK_HIGH_THRESHOLD   = float(os.getenv("RISK_HIGH_THRESHOLD",   0.20))

DEBUG_MODE            = os.getenv("DEBUG_MODE",            "True") == "True"
ANONYMIZE_PATIENT_IDS = os.getenv("ANONYMIZE_PATIENT_IDS", "True") == "True"
LOG_LEVEL             = os.getenv("LOG_LEVEL",             "INFO")

def validate_config():
    errors = []
    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_NEW_GROQ_KEY_HERE":
        errors.append("GROQ_API_KEY is not set in .env file")
    if BEAT_SIZE != 187:
        errors.append(f"BEAT_SIZE must be 187, got {BEAT_SIZE}")
    if MIT_BIH_SAMPLE_RATE != 360:
        errors.append(f"MIT_BIH_SAMPLE_RATE must be 360, got {MIT_BIH_SAMPLE_RATE}")
    if PTB_XL_SAMPLE_RATE != 500:
        errors.append(f"PTB_XL_SAMPLE_RATE must be 500, got {PTB_XL_SAMPLE_RATE}")
    if len(CLASS_NAMES) != 8:
        errors.append(f"CLASS_NAMES must have 8 classes, got {len(CLASS_NAMES)}")
    if errors:
        print("CONFIG WARNINGS:")
        for e in errors:
            print(f"  ⚠  {e}")
    else:
        print("Config loaded successfully")
    return errors

if __name__ == "__main__":
    print("=" * 60)
    print("ECG PROJECT CONFIG SUMMARY")
    print("=" * 60)
    print(f"Project Root           : {PROJECT_ROOT}")
    print(f"Data Root              : {DATA_ROOT}")
    print(f"MIT-BIH Raw            : {MIT_BIH_PATH}")
    print(f"MIT-BIH Processed      : {MIT_BIH_PROCESSED_PATH}")
    print(f"PTB-XL Raw             : {PTB_XL_PATH}")
    print(f"PTB-XL Processed       : {PTB_XL_PROCESSED_PATH}")
    print(f"Models Path            : {MODELS_PATH}")
    print(f"Checkpoints Path       : {CHECKPOINTS_PATH}")
    print(f"Beat Size              : {BEAT_SIZE}")
    print(f"MIT-BIH Sample Rate    : {MIT_BIH_SAMPLE_RATE} Hz")
    print(f"PTB-XL Sample Rate     : {PTB_XL_SAMPLE_RATE} Hz")
    print(f"Classes                : {CLASS_NAMES}")
    print(f"Debug Mode             : {DEBUG_MODE}")
    print(f"Groq Key Set           : {'YES' if GROQ_API_KEY and GROQ_API_KEY != 'PASTE_YOUR_NEW_GROQ_KEY_HERE' else 'NO'}")
    print("=" * 60)
    validate_config()