# ECG Arrhythmia Detection System

This project analyzes ECG signals to detect arrhythmias, flag abnormal beats, estimate session risk, generate reports, and expose the results through API and UI layers.

> Clinical disclaimer: This project is a decision-support system only. Medical conclusions must be reviewed by a qualified clinician.

## What This Project Includes

- `FastAPI` backend for prediction, reports, history, auth, and IoT routes
- Deep learning models for ECG beat classification and anomaly detection
- `Next.js` frontend for the web interface
- `Streamlit` dashboard for the legacy interactive workflow
- Database support with `SQLAlchemy` and `Alembic`

## Project Structure

```text
ecg_arrhythmia/
|-- api/          FastAPI application
|-- src/          ECG pipeline, models, agent, dashboard, storage
|-- frontend/     Next.js frontend
|-- tests/        Test files
|-- evaluation/   Evaluation scripts and results
|-- outputs/      Generated reports and outputs
|-- run.py        Starts backend and dashboard
```

## Requirements

- Python 3.11 recommended
- Node.js 18 or newer
- `pip`
- Trained model files referenced in `.env`

## Setup

### 1. Install Python Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Configure Environment Variables

Copy the template:

```bash
copy .env.template .env
```

Update the important values in `.env`:

- `GROQ_API_KEY`
- `PROJECT_ROOT`
- `DATA_ROOT`
- `MIT_BIH_PATH`
- `CNN_CLASSIFIER_PATH`
- `TRANSFORMER_AE_PATH`
- `DB_BACKEND`
- `DATABASE_URL` or `DB_PATH`
- `API_HOST`
- `API_PORT`
- `STREAMLIT_PORT`
- `REPORTS_PATH`

For the frontend, use:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Run The Project

### Run Backend And Streamlit Dashboard

```bash
python run.py
```

This starts:

- FastAPI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- Streamlit: `http://127.0.0.1:8501`

### Run Only The Backend

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Run The Frontend

```bash
cd frontend
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:3000
```

## Main API Routes

- `GET /health`
- `POST /predict`
- `POST /predict/single`
- `POST /agent/query`
- `POST /report/generate`
- `GET /report/download`
- `POST /signal/quality`
- `GET /history/{patient_id}`
- `POST /iot/stream`

Full API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Testing

Run the main test suite:

```bash
python tests/test_all.py
```

Or run all tests with:

```bash
pytest
```

## Docker

To start the containerized setup:

```bash
docker compose up --build
```

## Notes

- The repository includes both a legacy Streamlit dashboard and a newer Next.js frontend.
- Some default paths in `.env` point to local Windows directories, so update them before running on another machine.
