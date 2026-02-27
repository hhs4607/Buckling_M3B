# Tutorial 2: FastAPI & Backend — The Computation Server

> **For**: Engineers who know Python but not web servers.
> **Time**: ~15 minutes to read.

---

## What Problem Does This Solve?

The E3B analysis engine (`buckling_analysis_M3a.py`) uses numpy and must run in Python. Browsers cannot run Python. So we need a **Python server** that:

1. Listens for requests from the browser ("run this analysis with these parameters")
2. Executes the Python computation
3. Sends results back to the browser

**FastAPI** is the Python library that turns your functions into a web server.

---

## Core Concept 1: From Python Function to API Endpoint

Here is what the desktop GUI does when you click "Run":

```python
# Desktop GUI (simplified)
def run_full_analysis(self):
    vals = self.get_values()                    # Read from Tkinter entries
    mode = eval_m3_Pcr_and_mode(vals)           # Run computation
    P, dlin, dloc, dtot, dcr, _ = koiter_curves_from_mode(mode)
    self.display_full_results(results)          # Update Tkinter UI
```

Here is the equivalent as a FastAPI endpoint:

```python
# FastAPI backend (simplified)
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# 1. Define what the browser sends (request schema)
class BucklingRequest(BaseModel):
    core: str       # "m2" or "m3"
    params: dict    # {L: 1.5, b_root: 0.08, ...}

# 2. Define what we send back (response schema)
class BucklingResponse(BaseModel):
    Pcr: float
    dcr: float
    alpha_star: float
    # ... more fields

# 3. The endpoint — a decorated Python function
@router.post("/api/buckling/run")
async def run_buckling(request: BucklingRequest) -> BucklingResponse:
    # Same computation as desktop GUI!
    mode = eval_m3_Pcr_and_mode(request.params)
    P, dlin, dloc, dtot, dcr, _ = koiter_curves_from_mode(mode)

    return BucklingResponse(
        Pcr=mode["Pcr"],
        dcr=dcr,
        alpha_star=mode["alpha_star"],
        # ...
    )
```

**What changed**:
- Instead of reading from Tkinter widgets → parameters arrive as JSON in the HTTP request
- Instead of updating Tkinter labels → results go back as JSON in the HTTP response
- The actual computation (`eval_m3_Pcr_and_mode`) is **unchanged**

---

## Core Concept 2: HTTP Requests (How Browser Talks to Server)

HTTP is the protocol browsers use. Think of it as **function calls over the internet**:

```
Python function call:
    result = eval_m3_Pcr_and_mode(params)

HTTP equivalent:
    POST https://e3b-api.railway.app/api/buckling/run
    Body: {"core": "m3", "params": {"L": 1.5, "b_root": 0.08, ...}}

    Response: {"Pcr": 1234.56, "dcr": 0.0123, ...}
```

### HTTP Methods

| Method | Purpose | Analogy |
|--------|---------|---------|
| **GET** | Read data | Like calling a function with no side effects |
| **POST** | Send data + get result | Like calling a function that computes something |

Our API uses:
- `POST /api/buckling/run` — send params, get results
- `POST /api/sensitivity/run` — start sensitivity analysis
- `POST /api/sobol/run` — start Sobol analysis
- `GET /api/sensitivity/stream/{id}` — receive progress updates
- `GET /health` — check if server is alive

---

## Core Concept 3: Pydantic Validation (Automatic Input Checking)

In the desktop GUI, you manually validate inputs:

```python
# Desktop GUI validation (line 1065-1075)
def validate_inputs(self, vals):
    errors = []
    if vals["L"] <= 0: errors.append("Length must be > 0")
    if vals["b_root"] <= 0: errors.append("Root width must be > 0")
    if not (0 < vals["Vf"] < 1): errors.append("Vf must be 0 < Vf < 1")
```

With FastAPI + Pydantic, validation is **declarative** (built into the schema):

```python
from pydantic import BaseModel, Field, field_validator

class BucklingParams(BaseModel):
    L: float = Field(gt=0, description="Beam length in meters")
    b_root: float = Field(gt=0, description="Root width in meters")
    Vf: float = Field(gt=0, lt=1, description="Fiber volume fraction")
    nuf: float = Field(gt=-1, lt=0.5, description="Fiber Poisson ratio")
    face_angles: str = Field(description="Comma-separated ply angles")

    @field_validator('face_angles')
    @classmethod
    def validate_angles(cls, v: str) -> str:
        for part in v.split(','):
            float(part.strip())  # Raises ValueError if invalid
        return v
```

If someone sends invalid data, FastAPI **automatically** returns an error:

```
POST /api/buckling/run
Body: {"core": "m3", "params": {"L": -5, "Vf": 1.5}}

Response (422 Unprocessable Entity):
{
  "detail": [
    {"field": "L", "msg": "ensure this value is greater than 0"},
    {"field": "Vf", "msg": "ensure this value is less than 1"}
  ]
}
```

You do not write validation logic manually — Pydantic does it from the schema definition.

---

## Core Concept 4: SSE (Server-Sent Events) for Progress

The desktop GUI updates the progress bar from a worker thread:

```python
# Desktop GUI: progress update (line 1412-1417)
count += 1
progress = int(100 * count / total)
self.root.after(0, lambda p=progress:
    self.sens_progress.config(value=p)
)
```

On the web, the server **streams** progress events to the browser using SSE:

```python
# FastAPI: SSE streaming
from fastapi.responses import StreamingResponse
import json, asyncio

async def sensitivity_generator(job_params):
    """Yields SSE events as computation progresses."""
    total = calculate_total_evaluations(job_params)

    for i, result in enumerate(run_evaluations(job_params)):
        # Send progress event to browser
        progress_data = {
            "current": i + 1,
            "total": total,
            "message": f"Evaluating parameter {result['param']} ({i+1}/{total})"
        }
        yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"

    # Send final results
    yield f"event: result\ndata: {json.dumps(final_results)}\n\n"
    yield f"event: done\ndata: {{}}\n\n"

@router.get("/api/sensitivity/stream/{job_id}")
async def stream_sensitivity(job_id: str):
    return StreamingResponse(
        sensitivity_generator(jobs[job_id]),
        media_type="text/event-stream"
    )
```

### How SSE works visually:

```
Browser                              Server
   │                                    │
   │  GET /api/sensitivity/stream/abc   │
   │ ──────────────────────────────────▶│
   │                                    │  (starts computing)
   │  event: progress                   │
   │  data: {"current":1, "total":35}   │
   │ ◀──────────────────────────────────│
   │                                    │
   │  event: progress                   │
   │  data: {"current":2, "total":35}   │
   │ ◀──────────────────────────────────│
   │         ...                        │  (keeps sending)
   │                                    │
   │  event: result                     │
   │  data: {"results": [...]}          │
   │ ◀──────────────────────────────────│
   │                                    │
   │  event: done                       │
   │  data: {}                          │
   │ ◀──────────────────────────────────│
   │                                    │
   │  (connection closes)               │
```

The browser receives events **one by one** as they happen — the progress bar updates in real time, just like the desktop GUI.

---

## Core Concept 5: The Service Layer (Separation of Concerns)

We organize the backend into three layers:

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Router Layer (routers/buckling.py)                      │
│  - Receives HTTP request                                 │
│  - Validates input (Pydantic)                           │
│  - Calls service                                        │
│  - Returns HTTP response                                │
│  Analogy: The receptionist at a company                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Service Layer (services/buckling_service.py)            │
│  - Contains business logic                              │
│  - Coordinates computation                              │
│  - Converts numpy arrays → JSON-safe lists              │
│  - Generates contour grid data                          │
│  Analogy: The engineer who does the actual work         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Engine Layer (engine/buckling_analysis_M3a.py)         │
│  - The ORIGINAL computation code                        │
│  - eval_m2_Pcr(), eval_m3_Pcr_and_mode()               │
│  - koiter_curves_from_mode()                            │
│  - NO CHANGES from desktop version                      │
│  Analogy: The specialized machine that does the math    │
└─────────────────────────────────────────────────────────┘
```

Why separate layers?
- **Router**: Handles web stuff (HTTP, JSON, validation). If we change how the API works, only this layer changes.
- **Service**: Handles orchestration. If we add caching or logging, only this layer changes.
- **Engine**: Handles math. This is `buckling_analysis_M3a.py` — **never modified**.

---

## Core Concept 6: JSON (The Universal Data Format)

JSON is how the browser and server exchange data. It looks like a Python dictionary:

```python
# Python dict
results = {
    "Pcr": 1234.56,
    "dcr": 0.0123,
    "curves": {
        "P": [0.0, 100.0, 200.0, ...],
        "delta_total": [0.0, 0.001, 0.003, ...]
    }
}
```

```json
// JSON (exactly the same structure)
{
  "Pcr": 1234.56,
  "dcr": 0.0123,
  "curves": {
    "P": [0.0, 100.0, 200.0],
    "delta_total": [0.0, 0.001, 0.003]
  }
}
```

The key challenge: **numpy arrays cannot be sent as JSON**. They must be converted to Python lists first:

```python
# This fails:
json.dumps({"P": np.array([1.0, 2.0, 3.0])})  # TypeError!

# This works:
json.dumps({"P": np.array([1.0, 2.0, 3.0]).tolist()})  # OK
```

The service layer handles this conversion.

---

## Our API at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  POST /api/buckling/run                                      │
│  ├── Input:  {core, params}                                  │
│  └── Output: {Pcr, dcr, mode_params, curves, contour}       │
│       → Synchronous (returns in 0.5-5 seconds)              │
│                                                              │
│  POST /api/sensitivity/run                                   │
│  ├── Input:  {core, baseline_params, sweep_params}           │
│  └── Output: {job_id, total_evaluations, stream_url}        │
│       → Returns immediately, computation runs in background  │
│                                                              │
│  GET  /api/sensitivity/stream/{job_id}                       │
│  └── Output: SSE events (progress → result → done)          │
│       → Streams progress until computation finishes          │
│                                                              │
│  POST /api/sobol/run                                         │
│  ├── Input:  {core, baseline_params, uncertain_params, ...}  │
│  └── Output: {job_id, total_evaluations, stream_url}        │
│                                                              │
│  GET  /api/sobol/stream/{job_id}                             │
│  └── Output: SSE events (progress → result → done)          │
│                                                              │
│  POST /api/config/validate                                   │
│  ├── Input:  {params}                                        │
│  └── Output: {valid: true/false, errors: [...]}             │
│                                                              │
│  GET  /health                                                │
│  └── Output: {status: "ok"}                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## What You Need to Know for Design Review

1. **API shape**: Are the request/response fields correct? Missing anything?
   - "Should the buckling response also include the EI_total value?"
   - "Should sensitivity sweep support absolute ranges, not just ±%?"

2. **Computation fidelity**: The engine is unchanged, but the service layer converts data.
   - "Is the contour grid resolution (80×40) sufficient for web display?"

3. **Progress granularity**: How often should SSE send updates?
   - Currently: every single evaluation. For 1000 evaluations, that is 1000 events.

4. **Error handling**: What happens when computation fails?
   - "Should the API return partial results if one parameter in a sweep fails?"

---

## Summary

| Concept | One-line explanation |
|---------|---------------------|
| **FastAPI** | Python library that turns functions into web API endpoints |
| **Endpoint** | A URL + HTTP method that accepts and returns JSON data |
| **Pydantic** | Declarative input validation (like dataclass with built-in checks) |
| **SSE** | Server pushes real-time events to browser (for progress bars) |
| **Service layer** | Business logic between HTTP handling and computation engine |
| **JSON** | Universal data format (like Python dict) for browser-server communication |
| **ThreadPool** | Runs CPU-bound numpy code without blocking the web server |
