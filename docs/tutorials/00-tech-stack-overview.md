# E3B Web App — Technology Stack Overview

> **Audience**: Engineers and researchers with Python experience but no web development background.
> **Goal**: Understand each technology well enough to review the system design and suggest modifications.

---

## The Big Picture

Think of the E3B web app like a **client-server version of the desktop GUI**:

```
┌──────────────────────────────────────────────────────────────────┐
│                    DESKTOP (Current)                              │
│                                                                  │
│   gui_buckling_extended_FIXED.py                                 │
│   ┌────────────────────┬─────────────────────────────┐          │
│   │  Tkinter GUI       │  buckling_analysis_M3a.py   │          │
│   │  (User Interface)  │  (Computation Engine)        │          │
│   │                    │                              │          │
│   │  Same process,     │  numpy, matplotlib           │          │
│   │  same machine      │                              │          │
│   └────────────────────┴─────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────┘

                            ↓  becomes  ↓

┌──────────────────────────────────────────────────────────────────┐
│                      WEB (New)                                    │
│                                                                  │
│   ┌─────────────────────┐       ┌─────────────────────────────┐ │
│   │  Next.js Frontend   │  API  │  FastAPI Backend             │ │
│   │  (User Interface)   │◀─────▶│  (Computation Engine)       │ │
│   │                     │ HTTP  │                              │ │
│   │  Runs in BROWSER    │       │  Runs on SERVER              │ │
│   │  TypeScript, React  │       │  Python, numpy               │ │
│   │  Plotly.js charts   │       │  buckling_analysis_M3a.py   │ │
│   └─────────────────────┘       └─────────────────────────────┘ │
│        Vercel (cloud)              Railway (cloud)               │
└──────────────────────────────────────────────────────────────────┘
```

The key difference: the **user interface** and the **computation engine** now run on **different machines** and communicate over the internet.

---

## Technology Map

| Layer | Technology | Analogy (Python world) |
|-------|-----------|----------------------|
| **Frontend framework** | Next.js (React) | Like a "web Tkinter" — builds the UI |
| **UI components** | shadcn/ui | Pre-built widgets (buttons, inputs, tabs) — like ttk widgets |
| **Styling** | Tailwind CSS | Like CSS but with shorthand — replaces manual `.config()` calls |
| **Charts** | Plotly.js | Like matplotlib but interactive (zoom, hover) |
| **State management** | Zustand | Like Python variables that auto-update the UI when changed |
| **Language** | TypeScript | JavaScript with type hints (like Python type annotations) |
| **Backend framework** | FastAPI | Python web server — like Flask but faster, with validation |
| **Communication** | REST API + SSE | How frontend talks to backend (like function calls over HTTP) |
| **Frontend hosting** | Vercel | Cloud service that serves the website |
| **Backend hosting** | Railway | Cloud service that runs the Python server |
| **Math rendering** | KaTeX | Renders LaTeX equations in the browser |

---

## How They Fit Together

```
User opens browser
       │
       ▼
┌──────────────────┐
│  1. Next.js      │  Generates the HTML/CSS/JS that the browser displays
│     (React)      │  Components = reusable UI pieces (like Tkinter widgets)
│                  │
│  2. shadcn/ui    │  Pre-made components: Button, Input, Card, Tabs, etc.
│     + Tailwind   │  Tailwind = visual styling (colors, spacing, fonts)
│                  │
│  3. Zustand      │  Stores the current state (input values, results, etc.)
│                  │  When state changes → UI updates automatically
│                  │
│  4. Plotly.js    │  Renders interactive charts (load-deflection, contour)
│                  │
│  User clicks     │
│  "Run Analysis"  │
│       │          │
│       ▼          │
│  5. fetch() call │  Sends input data to the backend server (HTTP POST)
│       │          │
└───────┼──────────┘
        │  internet
        ▼
┌──────────────────┐
│  6. FastAPI      │  Receives the request, validates parameters
│                  │
│  7. Service      │  Calls the existing Python analysis functions
│     Layer        │  eval_m3_Pcr_and_mode(), koiter_curves_from_mode()
│                  │
│  8. Engine       │  buckling_analysis_M3a.py — UNCHANGED from desktop
│                  │  numpy does all the math
│                  │
│  9. Response     │  Sends results back as JSON (numbers + arrays)
│       │          │
└───────┼──────────┘
        │  internet
        ▼
┌──────────────────┐
│ 10. Zustand      │  Stores the results
│     updates      │
│       │          │
│       ▼          │
│ 11. Plotly.js    │  Renders charts from the result data
│     + React      │  ResultSummary shows Pcr, dcr, etc.
│                  │
│  User sees       │
│  results!        │
└──────────────────┘
```

---

## Tutorials (Read in Order)

| # | Tutorial | What You Will Learn |
|---|----------|-------------------|
| 1 | [Next.js & React](01-nextjs-react.md) | How the frontend is built, what components are, how pages work |
| 2 | [FastAPI & Backend](02-fastapi-backend.md) | How the Python server works, API endpoints, SSE streaming |
| 3 | [Frontend UI Stack](03-frontend-ui-stack.md) | Tailwind CSS, shadcn/ui, Plotly.js, Zustand — the visual layer |
| 4 | [Deployment](04-deployment.md) | Vercel + Railway, how the app goes from code to live website |

---

## Glossary

Quick reference for web terms you will encounter:

| Term | Meaning |
|------|---------|
| **Frontend** | The part that runs in the user's browser (HTML, CSS, JavaScript) |
| **Backend** | The part that runs on a server (Python, database, computation) |
| **API** | Application Programming Interface — how frontend talks to backend |
| **REST** | A style of API using HTTP methods (GET = read, POST = send data) |
| **JSON** | Data format like Python dict: `{"key": "value", "number": 42}` |
| **Component** | A reusable UI piece (e.g., a Button, an InputPanel, a Chart) |
| **State** | Current data in memory (e.g., which parameters the user entered) |
| **SSE** | Server-Sent Events — server pushes updates to browser (for progress bars) |
| **TypeScript** | JavaScript + type annotations (like `def f(x: int)` in Python) |
| **npm / pnpm** | Package managers for JavaScript (like pip for Python) |
| **Node.js** | JavaScript runtime for servers (like Python interpreter) |
| **Responsive** | UI adapts to screen size (desktop = side-by-side, mobile = stacked) |
| **Deploy** | Put code on a server so users can access it via URL |
| **Environment Variable** | Configuration stored outside code (like secrets, URLs) |
| **CORS** | Security rule: which websites can call your API |
| **Serverless** | Server that starts only when needed (Vercel uses this) |
| **Docker** | Packages your app + all dependencies into a portable container |
