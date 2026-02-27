# Tutorial 4: Deployment — Vercel + Railway

> **Audience**: Engineers and researchers who understand Python but not web deployment.
> **Goal**: Understand how the web app gets from your laptop to the internet, and why we use two services.

---

## Table of Contents

1. [The Core Concept: "Deploying" = Making Code Run on Someone Else's Computer](#1-the-core-concept)
2. [Why Two Services, Not One?](#2-why-two-services)
3. [Vercel — Frontend Hosting](#3-vercel--frontend-hosting)
4. [Railway — Backend Hosting](#4-railway--backend-hosting)
5. [Docker — Packaging the Backend](#5-docker--packaging-the-backend)
6. [Environment Variables — Connecting the Pieces](#6-environment-variables)
7. [The Request Journey — End to End](#7-the-request-journey)
8. [CORS — The Browser's Security Guard](#8-cors)
9. [Domains, URLs, and DNS](#9-domains-urls-and-dns)
10. [Deployment Workflow — From Code to Live](#10-deployment-workflow)
11. [Monitoring and Debugging](#11-monitoring-and-debugging)
12. [Cost and Scaling](#12-cost-and-scaling)
13. [Key Terminology](#13-key-terminology)

---

## 1. The Core Concept

### What "Deployment" Means

When you write a Python script, you run it on your laptop:

```
Your Laptop
┌──────────────────────────────────────┐
│  $ python gui_buckling_extended.py   │
│                                      │
│  → Opens window                      │
│  → You interact with it              │
│  → Only you can use it               │
└──────────────────────────────────────┘
```

**Deployment** means uploading your code to a computer in a data center (a "server") so anyone with a web browser can use it:

```
Your Laptop                  Cloud Server                User's Browser
┌────────────┐   upload    ┌──────────────┐   internet  ┌──────────────┐
│ Your code  │ ──────────▶ │ Runs 24/7    │ ◀─────────▶ │ Anyone can   │
│            │             │ Has a URL    │             │ access it    │
└────────────┘             └──────────────┘             └──────────────┘
```

### The Python Analogy

| Concept | Python World | Web World |
|---------|-------------|-----------|
| Run code | `python script.py` | Deploy to cloud server |
| Share code | "Here's the .py file" | "Here's the URL" |
| Dependencies | `pip install -r requirements.txt` | Docker builds everything |
| Environment | Your laptop's Python | Cloud server's container |

---

## 2. Why Two Services?

### The Problem with One Server

You might ask: "Why not just put everything on one server?"

We could, but two specialized services work better for our case:

```
Option A: Single Server (VPS)                     Option B: Two Services (Our Choice)
┌─────────────────────────────┐                   ┌─────────────────────┐
│  Both frontend + backend    │                   │  Vercel (Frontend)  │
│                             │                   │  Free, fast, global │
│  - You manage everything    │                   │  Auto-HTTPS, CDN    │
│  - $5-20/month minimum      │                   │  Zero config        │
│  - Manual HTTPS setup       │                   └─────────────────────┘
│  - Manual scaling           │                           +
│  - You fix crashes          │                   ┌─────────────────────┐
└─────────────────────────────┘                   │  Railway (Backend)  │
                                                  │  Docker, long-lived │
                                                  │  SSE support        │
                                                  └─────────────────────┘
```

### Why Vercel for Frontend?

Vercel is built specifically for Next.js (they created Next.js). It provides:

- **Zero configuration**: Connect your GitHub repo, done
- **Global CDN**: Your files are copied to data centers worldwide — users load from the nearest one
- **Automatic HTTPS**: Security certificate handled for you
- **Instant deploys**: Push to GitHub → live in seconds
- **Free tier**: More than enough for our usage

Think of it like **GitHub Pages on steroids** — it hosts static files and runs small server-side functions.

### Why Railway for Backend?

Railway runs Docker containers, which is perfect for our Python backend because:

- **Long-running processes**: Vercel's serverless functions time out after 10-60 seconds. Our Sobol analysis can run for minutes. Railway keeps the process alive.
- **Persistent connections**: SSE (progress streaming) needs a connection that stays open. Vercel would kill it.
- **Full Python environment**: numpy, scipy, pandas — all run natively, not in a constrained serverless sandbox.
- **Docker support**: Package everything exactly as needed.

### The Timeout Problem (Key Reason for Split)

```
Vercel Serverless Function:
┌──────────────────────────────────────────────┐
│  Request comes in                            │
│  │                                           │
│  ▼                                           │
│  Function starts                             │
│  │                                           │
│  │ ... doing work ...                        │
│  │                                           │
│  ▼                                           │
│  ⏰ 10 seconds (free) / 60 seconds (pro)     │
│  │                                           │
│  ▼                                           │
│  KILLED! "Function timed out"                │
│                                              │
│  Our Sobol analysis: 1-10+ minutes  → ❌     │
└──────────────────────────────────────────────┘

Railway Container:
┌──────────────────────────────────────────────┐
│  Container starts and runs indefinitely      │
│  │                                           │
│  ▼                                           │
│  Request comes in                            │
│  │                                           │
│  │ ... computing for 5 minutes ...           │
│  │ ... streaming progress via SSE ...        │
│  │                                           │
│  ▼                                           │
│  Done! Response sent.  → ✅                   │
│                                              │
│  Container keeps running for next request    │
└──────────────────────────────────────────────┘
```

---

## 3. Vercel — Frontend Hosting

### What Vercel Does

Vercel takes your Next.js project and does three things:

1. **Builds** it (compiles TypeScript, bundles CSS, optimizes images)
2. **Serves static files** via a global CDN (HTML, CSS, JS, images)
3. **Runs API routes** as serverless functions (for short-lived requests)

### How It Works (Simplified)

```
Your Code (GitHub)          Vercel Build                  User's Browser
┌───────────────┐          ┌──────────────┐              ┌──────────────┐
│ src/           │  push   │ npm run build│  CDN deploy  │ GET /         │
│   app/         │ ──────▶ │              │ ──────────▶  │ ◀── HTML/JS  │
│   components/  │         │ Compiles TS  │              │              │
│   lib/         │         │ Bundles CSS  │              │ GET /analysis │
│                │         │ Optimizes    │              │ ◀── HTML/JS  │
└───────────────┘          └──────────────┘              └──────────────┘
```

### The CDN Concept

CDN = Content Delivery Network. Your files are copied to ~100 locations worldwide:

```
Without CDN:                          With CDN (Vercel):

User in Korea ─── 12,000 km ──▶ US    User in Korea ──▶ Seoul Edge ✓
User in Brazil ── 8,000 km ──▶ US     User in Brazil ──▶ São Paulo Edge ✓
User in UK ────── 5,000 km ──▶ US     User in UK ────▶ London Edge ✓

All users wait for long round-trip    Each user gets files from nearby
```

### Serverless Functions

Vercel also runs small server-side logic. Our Next.js API proxy (`/api/proxy/...`) runs as a serverless function:

```
Browser                  Vercel Serverless              Railway Backend
┌──────────┐            ┌───────────────┐              ┌──────────────┐
│ Click    │  POST      │ API route     │  Forward     │ FastAPI      │
│ "Run"    │ ─────────▶ │ /api/proxy/   │ ───────────▶ │ /api/bucket/ │
│          │            │ buckling/run  │              │ run          │
│          │ ◀───────── │               │ ◀─────────── │              │
│ Show     │  JSON      │ (< 10 sec)   │  JSON        │ Computes...  │
│ results  │            └───────────────┘              └──────────────┘
└──────────┘
```

This works for the buckling single-case analysis (~2-5 seconds). But for long-running SSE streams, the browser connects directly to Railway.

### Vercel Dashboard

When you deploy, Vercel provides:
- **Deployment URL**: `https://e3b-buckling.vercel.app`
- **Build logs**: See if build succeeded or failed
- **Analytics**: Page views, performance metrics
- **Environment variables**: Set `NEXT_PUBLIC_API_URL`

---

## 4. Railway — Backend Hosting

### What Railway Does

Railway takes a Docker container (or a GitHub repo) and runs it as a persistent service:

```
Your Code (GitHub)         Railway                       Internet
┌───────────────┐         ┌─────────────────┐           ┌──────────┐
│ backend/      │  push   │ Build Docker    │  run      │ API      │
│   app/        │ ──────▶ │ image           │ ──────▶   │ available│
│   Dockerfile  │         │                 │           │ 24/7     │
│   requirements│         │ Start container │           │          │
└───────────────┘         └─────────────────┘           └──────────┘
```

### How Railway Differs from Vercel

| Feature | Vercel (Serverless) | Railway (Container) |
|---------|-------------------|-------------------|
| Execution model | Function per request | Long-running process |
| Timeout | 10-60 seconds | None (runs forever) |
| State between requests | None (stateless) | In-memory (dict, list) |
| Startup | Cold start per request | Container always running |
| Best for | Static sites, short APIs | Long computations, streams |
| Python support | Limited (Lambda) | Full Docker environment |

### Railway's Pricing Model

Railway charges for **actual usage**:
- CPU time consumed
- Memory used
- Network transfer

For our case (<10 users, intermittent computation), this falls well within the free tier ($5/month credit).

### Cold Starts

When nobody uses the app for ~15 minutes, Railway may "sleep" the container to save resources:

```
Timeline:
─────────────────────────────────────────────────────
                                     Container sleeps
User A clicks "Run"                       │
│                                         │
▼                                         ▼
Request → Container starts (5-10 sec) → Responds
          (cold start)

Next request (within 15 min):
User B clicks "Run"
│
▼
Request → Already running → Responds immediately
          (warm)
```

Mitigation: The frontend shows "Connecting to computation server..." during the first slow request.

---

## 5. Docker — Packaging the Backend

### What Docker Is

Docker is like a **virtual machine, but lighter**. It packages your code + all dependencies into a single "image" that runs identically everywhere.

**The Python analogy**: Like `venv` + `requirements.txt`, but it also includes the operating system, Python itself, and system libraries. Guaranteed identical behavior.

```
Without Docker:                       With Docker:

"Works on my machine" ≠              "Works on my machine" =
"Works on server"                    "Works on server"

Different Python versions,           Same everything,
different numpy versions,            every time,
missing system libraries...          everywhere.
```

### Our Dockerfile (Simplified)

```dockerfile
# Start from official Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our application code
COPY app/ ./app/

# Tell Docker what command starts our server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### How Docker Build Works

```
Step by step:

1. FROM python:3.12-slim
   └─ Downloads a minimal Linux + Python 3.12 image (~150MB)

2. COPY requirements.txt .
   └─ Copies just the requirements file into the container

3. RUN pip install -r requirements.txt
   └─ Installs: fastapi, uvicorn, numpy, pandas, pydantic
   └─ This is cached! Only re-runs if requirements.txt changes.

4. COPY app/ ./app/
   └─ Copies our FastAPI application code

5. CMD ["uvicorn", ...]
   └─ Defines the startup command

Result: A self-contained image (~500MB) with everything needed.
```

### Docker Layer Caching

Docker is smart about rebuilding. Each step is a "layer":

```
requirements.txt unchanged?
│
├─ YES → Reuse cached layer (fast, ~0 seconds)
│
└─ NO  → Rebuild from this step onwards

app/ code changed?
│
├─ YES → Only rebuild the COPY app/ step (fast, ~2 seconds)
│        (requirements layer is still cached)
│
└─ NO  → Everything cached, nothing to do
```

This is why `requirements.txt` is copied before `app/` — changing code does not reinstall packages.

---

## 6. Environment Variables

### What They Are

Environment variables are **configuration values** that change between environments (your laptop vs. production) but should not be hardcoded in source code.

**Python analogy**: Like `os.environ['MY_VAR']` — values that exist outside your code.

### Our Environment Variables

```
┌─────────────────────────────────────────────────────────┐
│  Vercel (Frontend)                                      │
│                                                         │
│  NEXT_PUBLIC_API_URL = https://e3b-api.railway.app      │
│                                                         │
│  → Tells the frontend WHERE the backend lives           │
│  → "NEXT_PUBLIC_" prefix makes it available in browser  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Railway (Backend)                                      │
│                                                         │
│  ALLOWED_ORIGINS = https://e3b-buckling.vercel.app      │
│                                                         │
│  → Tells the backend WHICH frontends can call it        │
│  → Security: only our Vercel app, not random websites   │
└─────────────────────────────────────────────────────────┘
```

### Why Not Hardcode URLs?

Because the URLs differ between environments:

```
Development (your laptop):
  Frontend: http://localhost:3000
  Backend:  http://localhost:8000
  NEXT_PUBLIC_API_URL = http://localhost:8000

Production (live):
  Frontend: https://e3b-buckling.vercel.app
  Backend:  https://e3b-api.railway.app
  NEXT_PUBLIC_API_URL = https://e3b-api.railway.app
```

Same code, different configuration. Environment variables make this seamless.

### The `NEXT_PUBLIC_` Prefix

In Next.js, environment variables starting with `NEXT_PUBLIC_` are embedded into the browser JavaScript at build time. Other variables are only available on the server.

```
NEXT_PUBLIC_API_URL → Available in browser code ✅ (needed for API calls)
DATABASE_PASSWORD   → Only on server ✅ (secrets stay safe)
```

---

## 7. The Request Journey

### Short Request (Buckling Single Case)

```
User's Browser           Vercel CDN          Vercel Function       Railway
┌──────────┐            ┌──────────┐        ┌──────────────┐     ┌──────────┐
│ 1. Opens │  GET /     │ 2. Serve │        │              │     │          │
│    e3b-  │ ─────────▶ │    HTML, │        │              │     │          │
│    buckling│           │    CSS,  │        │              │     │          │
│    .vercel│           │    JS    │        │              │     │          │
│    .app  │ ◀───────── │    files │        │              │     │          │
│          │            └──────────┘        │              │     │          │
│          │                                │              │     │          │
│ 3. Click │  POST /api/proxy/buckling/run  │ 4. Forward   │     │ 5.Compute│
│   "Run"  │ ──────────────────────────────▶│    request   │────▶│   Pcr    │
│          │                                │    to Railway│     │          │
│ 7. Show  │ ◀──────────────────────────────│ 6. Return   │◀────│   Done!  │
│  results │            JSON response       │    JSON      │     │          │
└──────────┘                                └──────────────┘     └──────────┘

Total time: ~2-5 seconds
```

### Long Request (Sobol Analysis — Direct to Railway)

```
User's Browser                                        Railway
┌──────────────┐                                     ┌──────────────┐
│ 1. Click     │  POST /api/sobol/run                │ 2. Start job │
│    "Run      │ ───────────────────────────────────▶ │    Return    │
│     Sobol"   │ ◀─────────────────────────────────── │    job_id    │
│              │         { "job_id": "abc-123" }     │              │
│              │                                      │              │
│ 3. Open SSE  │  GET /api/sobol/stream/abc-123      │ 4. Stream    │
│    connection│ ───────────────────────────────────▶ │    progress  │
│              │                                      │              │
│              │ ◀─ data: {"progress": 0.05, ...}    │  Computing   │
│  Update bar  │ ◀─ data: {"progress": 0.10, ...}    │  ...         │
│  5%... 10%.. │ ◀─ data: {"progress": 0.50, ...}    │  ...         │
│  50%...      │ ◀─ data: {"progress": 1.00, ...}    │              │
│              │ ◀─ data: {"type": "result", S1, ST}  │  Done!       │
│              │ ◀─ event: done                       │              │
│ 5. Show chart│                                      │              │
└──────────────┘                                     └──────────────┘

Note: This bypasses Vercel entirely! Browser → Railway directly.
Total time: 1-10+ minutes (shown with live progress bar)
```

---

## 8. CORS — The Browser's Security Guard

### The Problem CORS Solves

Browsers have a security rule: JavaScript on `site-a.com` cannot make requests to `site-b.com` unless `site-b.com` explicitly allows it.

```
Without CORS:
  Malicious site: evil.com
  ┌─────────────────────────────────────────────┐
  │ <script>                                    │
  │   fetch("https://your-bank.com/transfer",   │
  │         { body: "send $1000 to hacker" })   │
  │ </script>                                   │
  └─────────────────────────────────────────────┘
  Browser: "BLOCKED! your-bank.com didn't allow requests from evil.com"
```

### How It Applies to Us

Our frontend is on `e3b-buckling.vercel.app` and our backend is on `e3b-api.railway.app`. These are **different origins** (different domain names), so the browser blocks requests by default.

```
e3b-buckling.vercel.app                    e3b-api.railway.app
┌─────────────────────┐                   ┌───────────────────┐
│ fetch("/api/run")   │   CORS check      │                   │
│                     │ ────────────────▶  │ Is vercel.app     │
│                     │                    │ in ALLOWED_ORIGINS?│
│                     │  Yes, allowed!     │                   │
│                     │ ◀────────────────  │ Yes → proceed     │
│  ✅ Request works    │                    │                   │
└─────────────────────┘                   └───────────────────┘
```

### Our CORS Configuration (FastAPI)

```python
# In FastAPI backend (app/main.py)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://e3b-buckling.vercel.app",  # Production frontend
        "http://localhost:3000",              # Local development
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

This tells the browser: "Requests from these URLs are OK."

---

## 9. Domains, URLs, and DNS

### URL Anatomy

```
https://e3b-api.railway.app/api/buckling/run
│       │                    │
│       │                    └─ Path (which endpoint)
│       └─ Domain (which server)
└─ Protocol (secure HTTP)
```

### How DNS Works (Simplified)

DNS = Domain Name System. It converts human-readable names to IP addresses.

```
Your Browser                 DNS Server                    Railway Server
┌──────────┐                ┌──────────┐                  ┌──────────┐
│ Navigate │  "Where is     │          │                  │          │
│ to e3b-  │  e3b-api.     │ It's at  │                  │ IP:      │
│ api.     │  railway.app?"│ 34.56.   │                  │ 34.56.   │
│ railway  │ ────────────▶ │ 78.90    │                  │ 78.90    │
│ .app     │ ◀──────────── │          │                  │          │
│          │                └──────────┘                  │          │
│          │  Connect to 34.56.78.90                      │          │
│          │ ───────────────────────────────────────────▶ │          │
└──────────┘                                             └──────────┘
```

### Our URLs

| Environment | Frontend URL | Backend URL |
|-------------|-------------|------------|
| Development | `http://localhost:3000` | `http://localhost:8000` |
| Production | `https://e3b-buckling.vercel.app` | `https://e3b-api.railway.app` |
| Custom domain (optional) | `https://e3b.yourdomain.com` | `https://api.e3b.yourdomain.com` |

Both Vercel and Railway provide free `*.vercel.app` and `*.railway.app` subdomains. Custom domains can be added later if desired.

---

## 10. Deployment Workflow

### Initial Setup (One-Time)

```
1. Push code to GitHub
   $ git push origin main

2. Connect Vercel to GitHub repo
   → Vercel Dashboard → "New Project" → Select repo
   → Set root directory: frontend/
   → Set NEXT_PUBLIC_API_URL environment variable
   → Click "Deploy"

3. Connect Railway to GitHub repo
   → Railway Dashboard → "New Project" → Select repo
   → Set root directory: backend/
   → Set ALLOWED_ORIGINS environment variable
   → Railway detects Dockerfile, builds automatically
```

### Ongoing Deployments (Every Code Change)

```
Developer                GitHub                Vercel              Railway
┌──────────┐            ┌──────────┐          ┌──────────┐       ┌──────────┐
│ Edit code│  git push  │ Webhook  │ trigger  │ Build    │       │          │
│          │ ─────────▶ │ notifies │ ───────▶ │ frontend │       │          │
│          │            │ both     │          │ Deploy!  │       │          │
│          │            │ services │ ───────▶ │          │       │ Build    │
│          │            │          │ trigger  │          │       │ backend  │
│          │            │          │          │          │       │ Deploy!  │
└──────────┘            └──────────┘          └──────────┘       └──────────┘

Total time: ~1-3 minutes from push to live.
```

### Preview Deployments (Vercel Feature)

When you create a **pull request** on GitHub, Vercel automatically creates a preview deployment:

```
main branch     → https://e3b-buckling.vercel.app         (production)
feature branch  → https://e3b-buckling-abc123.vercel.app   (preview)
```

This lets you test changes before merging to production. Very useful for reviewing UI changes.

---

## 11. Monitoring and Debugging

### Vercel Monitoring

Vercel provides:
- **Build logs**: See compilation output, catch build errors
- **Function logs**: See API route execution and errors
- **Analytics**: Page load times, visitor counts
- **Error tracking**: Runtime errors in serverless functions

```
Example build log:
  ▲ Building Next.js...
  ✓ Compiled successfully
  ✓ Collecting page data
  ✓ Generating static pages (5/5)
  ✓ Collecting build traces

  Route (app)                  Size     First Load JS
  ┌ ○ /                        5.2 kB   89.3 kB
  ├ ○ /analysis                12.1 kB  96.2 kB
  └ ○ /manual                  3.1 kB   87.2 kB
```

### Railway Monitoring

Railway provides:
- **Deploy logs**: Docker build output
- **Runtime logs**: stdout/stderr from your FastAPI app (like terminal output)
- **Metrics**: CPU, memory, network usage
- **Health checks**: Automatic ping to verify the service is alive

```
Example runtime log:
  INFO:     Started server process [1]
  INFO:     Waiting for application startup.
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000
  INFO:     192.168.1.1 - "POST /api/buckling/run" 200 OK
  INFO:     192.168.1.1 - "GET /api/sobol/stream/abc-123" 200 OK
```

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Build fails on Vercel | Red "X" in dashboard | Check build logs for TypeScript errors |
| Backend unreachable | "Network error" in browser | Check Railway logs, verify ALLOWED_ORIGINS |
| CORS error | Console shows "CORS policy" | Add frontend URL to ALLOWED_ORIGINS |
| Cold start slow | First request takes 10 seconds | Add health check, show loading message |
| SSE disconnects | Progress bar stops updating | Check Railway connection limits, add reconnect logic |
| Wrong API URL | "404 Not Found" | Verify NEXT_PUBLIC_API_URL matches Railway URL |

---

## 12. Cost and Scaling

### Free Tier Comparison

| Feature | Vercel Free | Railway Free ($5/mo credit) |
|---------|-------------|---------------------------|
| Bandwidth | 100 GB/month | Included |
| Build minutes | 6000 min/month | Included |
| Serverless execution | 100 GB-hours | N/A |
| Container hours | N/A | ~500 hours/month |
| Custom domains | Yes | Yes |
| HTTPS | Automatic | Automatic |

For our use case (<10 users, intermittent computation), both free tiers are more than sufficient.

### What Happens Under Load?

```
<10 Users (Our case):
  Vercel:  One serverless function at a time → fine
  Railway: Single container, 2-4 thread pool → fine

10-50 Users:
  Vercel:  Auto-scales serverless functions → still fine
  Railway: May need to increase container resources → $5-10/month

50+ Users:
  Vercel:  Still fine (CDN handles static files)
  Railway: Multiple container instances needed → more complex setup
  → At this point, consider a dedicated VPS or cloud solution
```

### Why This Architecture Scales Enough

The key insight: most of the "load" is serving HTML/CSS/JS files, which Vercel's CDN handles effortlessly. The backend only gets called when a user clicks "Run" — not on every page view. With <10 users who run analyses intermittently, the backend load is negligible.

---

## 13. Key Terminology

| Term | Definition | Analogy |
|------|-----------|---------|
| **Deploy** | Upload code to run on a cloud server | "Installing" your program on a remote computer |
| **CDN** | Content Delivery Network — copies files to servers worldwide | Having copies of a book in every local library |
| **Serverless** | Cloud runs your function on-demand, you don't manage servers | Renting a taxi vs owning a car |
| **Container** | Isolated environment with your code + dependencies | A virtual machine, but lightweight |
| **Docker** | Tool for building and running containers | Like `venv` + `requirements.txt` but for the whole OS |
| **Dockerfile** | Recipe for building a Docker image | Like a `Makefile` but for containers |
| **Image** | A built, ready-to-run container template | Like a `.iso` file for an OS |
| **Environment Variable** | Configuration value set outside code | Like `sys.argv` but persistent |
| **CORS** | Browser security rule for cross-origin requests | A bouncer checking your ID at the door |
| **HTTPS** | Encrypted HTTP (the "s" = secure) | Sending a sealed letter vs a postcard |
| **DNS** | Translates domain names to IP addresses | A phone book for the internet |
| **Health Check** | Periodic ping to verify a service is alive | "Are you still there?" every 30 seconds |
| **Cold Start** | Delay when a sleeping service wakes up | Warming up a cold engine |
| **Webhook** | Automatic notification from one service to another | "Notify me when code is pushed" |
| **Preview Deploy** | Temporary deployment for testing a branch | Staging server per pull request |
| **Origin** | Protocol + domain + port (e.g., `https://example.com:443`) | A website's "identity" for CORS |
| **Proxy** | Intermediary that forwards requests | A middleman passing messages |

---

## Summary: The Complete Picture

```
Developer                                              Users
┌──────────┐                                          ┌──────────┐
│ git push │                                          │ Browser  │
└────┬─────┘                                          └────┬─────┘
     │                                                     │
     ▼                                                     │
┌──────────┐                                               │
│ GitHub   │── webhook ──┐                                 │
└──────────┘             │                                 │
                    ┌────┴────┐                             │
                    ▼         ▼                             │
             ┌──────────┐ ┌──────────┐                     │
             │ Vercel   │ │ Railway  │                     │
             │ builds   │ │ builds   │                     │
             │ frontend │ │ Docker   │                     │
             └────┬─────┘ └────┬─────┘                     │
                  │            │                            │
                  ▼            ▼                            │
          ┌─────────────┐ ┌──────────────┐                 │
          │ Vercel CDN  │ │ Railway      │                 │
          │ HTML/CSS/JS │ │ Container    │                 │
          │ + API proxy │ │ FastAPI+numpy│                 │
          └──────┬──────┘ └──────┬───────┘                 │
                 │               │                          │
                 │   ┌───────────┘                          │
                 │   │                                      │
                 ▼   ▼                                      │
          ┌─────────────────┐                               │
          │ Internet        │ ◀─────────────────────────────┘
          │ (HTTPS + CORS)  │
          └─────────────────┘
```

The deployment architecture may seem complex at first, but each piece has a clear purpose:
- **Vercel** serves the user interface fast and globally
- **Railway** runs the Python computation engine without timeouts
- **Docker** ensures the backend runs identically everywhere
- **Environment variables** connect the pieces without hardcoding
- **CORS** keeps the communication secure
- **GitHub webhooks** automate everything after a code push
