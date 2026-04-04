# Frontend

This folder is the production migration path away from Streamlit.

Planned stack:
- Next.js app router
- TypeScript
- FastAPI as the only backend contract

Expected environment variable:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Typical commands:

```bash
npm install
npm run dev
```

Initial scope in this scaffold:
- health/status integration
- patient history shell
- dashboard layout for future prediction/report flows
