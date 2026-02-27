## Why

The E3B Buckling Analysis tool currently exists only as a desktop Tkinter GUI (`gui_buckling_extended_FIXED.py`), limiting accessibility to users who can install Python locally. Converting it to a web application makes the tool publicly accessible via a browser, eliminates installation friction, enables mobile usage, and provides a modern, professional interface befitting Embraer's engineering tooling. The user base is small (<10 concurrent users) but geographically distributed.

## What Changes

- **New web frontend**: Next.js 15 application with responsive UI covering all 3 analysis modes (Buckling, Sensitivity, Uncertainty)
- **New API backend**: FastAPI service wrapping the existing `buckling_analysis_M3a.py` computation engine (unchanged)
- **New manual pages**: Theory Manual and User Manual converted from PDF to structured HTML with KaTeX equation rendering
- **New landing page**: Public-facing introduction to the tool with feature overview
- **New deployment infrastructure**: Vercel (frontend) + Railway (backend) split deployment
- **Interactive charts**: Replace static matplotlib plots with interactive Plotly.js (zoom, pan, hover, export)
- **Real-time progress**: SSE-based progress streaming for long-running Sensitivity and Sobol analyses
- **JSON config management**: Browser-based save/load of analysis configurations
- **Input validation**: Real-time field validation with visual feedback (matching desktop GUI rules)
- Desktop GUI (`gui_buckling_extended_FIXED.py`) remains untouched for backward compatibility

## Capabilities

### New Capabilities
- `computation-api`: FastAPI backend exposing buckling/sensitivity/sobol analysis as REST endpoints with SSE progress streaming. Wraps existing Python engine without modification.
- `analysis-ui`: Next.js frontend with 3-tab analysis interface (Buckling, Sensitivity, Uncertainty), interactive Plotly.js charts, input validation, config save/load, and responsive mobile layout.
- `manual-pages`: Theory Manual and User Manual rendered as structured HTML pages with KaTeX equations, sidebar TOC navigation, and responsive layout.
- `deployment`: Split deployment configuration — Vercel for Next.js frontend, Railway (Docker) for FastAPI backend, with CORS and API proxy setup.

### Modified Capabilities
(none — this is a new web application; the existing desktop GUI and analysis engine are not modified)

## Impact

- **New code**: ~40 frontend components (TypeScript/React), ~10 backend modules (Python/FastAPI), 2 manual pages
- **Existing code**: `buckling_analysis_M3a.py` is copied into backend as-is; no modifications
- **Dependencies**: Node.js ecosystem (Next.js, Plotly.js, Zustand, shadcn/ui, Tailwind CSS) + Python (FastAPI, uvicorn, pydantic)
- **Infrastructure**: Two new cloud services (Vercel free tier + Railway free/starter tier)
- **APIs**: 4 new REST endpoints + 2 SSE stream endpoints
- **No breaking changes**: Desktop GUI continues to work independently
