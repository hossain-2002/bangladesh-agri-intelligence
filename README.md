# Bangladesh Agriculture Intelligence

Bangla-first, offline-first agriculture intelligence for Bangladeshi smallholders with plots under 2 hectares.

A farmer photographs a sick crop and gets a diagnosis in seconds, asks a Bangla voice advisor questions, and receives flood, rain and pest warnings for their own plot.
The farmer also follows a simple seasonal plan for rice, potato, vegetables and jute.

## Who this is for

Smallholder farmers growing rice (aman and boro), potato, vegetables (brinjal / tomato) and jute, using low-to-mid Android phones with intermittent 2G/3G/4G.

No proprietary hardware: phone camera + free Sentinel-2 satellite + public weather only.

## Tech stack

- Backend: Python 3.12 + FastAPI (modular monolith)
- Frontend: React + Vite + TypeScript, PWA (offline-first)
- AI: Google Gemini API (planned: Flash by default, Pro for escalated cases)
- Offline diagnosis: TensorFlow Lite classifier, on device
- Database: PostgreSQL 16 + PostGIS
- Cache / queue: Redis
- Satellite: Sentinel-2 via Copernicus (free, 10 m)
- Hosting (pilot): Google Cloud Run + Cloud SQL, or a single VPS

Full design: see `docs/` and the 21-page MVP blueprint.

## Project structure

    backend/
        FastAPI API, modules, jobs, Alembic migrations
    frontend/
        React PWA (farmer, partner and admin routes)
    ml/
        Leaf-disease classifier training and TFLite export
    docs/
        Architecture notes, decisions and launch checklists

## Local setup (Windows, PowerShell)

Prerequisites:

- Python 3.12
- Node.js
- PostgreSQL
- Docker (optional)

The folders are empty in this initial commit. The commands below are the setup you will use once Phase 0 adds the backend and frontend skeletons.

### Step 1. Clone the repo and copy the environment file

    git clone https://github.com/hossain-2002/bangladesh-agri-intelligence.git
    cd bangladesh-agri-intelligence
    Copy-Item .env.example .env

Fill in your own keys in `.env`. Never commit `.env`.

### Step 2. Set up the backend

    cd backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

### Step 3. Set up the frontend

    cd frontend
    npm install
    npm run dev

## Safety note

Pesticide and fertiliser advice ships only from agronomist-approved content, with safety warnings. The LLM phrases approved advice; it does not invent dosages.

Agronomy content must be reviewed by a qualified local agronomist before advice reaches farmers.

## Status

Initial commit only: README, `.gitignore`, `.env.example` and folder structure. No application features implemented yet.
