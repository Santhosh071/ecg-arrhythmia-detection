import sys
import json
import os

sys.path.insert(0, r"C:\ecg_arrhythmia")

from dotenv import load_dotenv
load_dotenv()

from src.layer4_agent.agent_tools import ALL_TOOLS, init_tools
from src.layer4_agent import ECGDatabase

db = ECGDatabase(os.getenv("DB_PATH", r"D:\ecg_project\database\patient_history.db"))
init_tools(db, None)

result = ALL_TOOLS[3].run(json.dumps({
    "patient_id": "TEST_001",
    "risk_level": "Critical",
    "message"   : "Test alert - PVC detected",
    "class_name": "V",
    "beat_index": 42
}))

print("Result:", result)
print("\nCheck this folder for the alert file:")
print(r"  C:\ecg_arrhythmia\outputs\alerts")