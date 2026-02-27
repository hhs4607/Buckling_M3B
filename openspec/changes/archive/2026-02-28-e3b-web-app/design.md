## Context

The E3B Buckling Analysis tool is a Python desktop application (Tkinter GUI + numpy computation engine) that performs local buckling analysis of double-tapered composite box beams. The computation engine (`buckling_analysis_M3a.py`, 698 lines) implements micromechanics, CLT, Ritz energy minimization, and Koiter post-buckling analysis. The GUI (`gui_buckling_extended_FIXED.py`, 1769 lines) provides 3 analysis modes: single-case Buckling, OAT Sensitivity, and Sobol Uncertainty Quantification.

The computation engine is CPU-bound (numpy array operations) and takes 0.5-5 seconds per evaluation depending on solver (M2 fast vs M3 accurate) and grid resolution. Sensitivity/Sobol analyses run hundreds of evaluations sequentially.

Key constraint: The computation engine must remain in Python (numpy dependency). The frontend must be accessible via public URL with no installation.

## Goals / Non-Goals

**Goals:**
- Expose all 3 analysis modes (Buckling, Sensitivity, Sobol UQ) via web browser
- Provide responsive UI that works on desktop and mobile devices
- Show real-time progress for long-running analyses (Sensitivity, Sobol)
- Produce interactive charts (zoom, pan, hover, export) replacing static matplotlib
- Serve Theory and User Manuals as navigable HTML pages with equation rendering
- Deploy with minimal operational cost (free/low-cost tiers, <10 users)
- Preserve numerical accuracy — web results must match desktop GUI exactly

**Non-Goals:**
- User authentication or access control (public tool)
- Multi-user collaboration or shared sessions
- Modifying the computation engine (`buckling_analysis_M3a.py`)
- Replacing the desktop GUI (it remains functional independently)
- Database or persistent storage (all computation is stateless)
- Parallel/distributed computation (single-threaded numpy is sufficient)
- Internationalization beyond English (engineering tool)

## Decisions

### 1. Split Deployment: Vercel (Frontend) + Railway (Backend)

**Decision**: Deploy Next.js frontend on Vercel and FastAPI backend on Railway as separate services.

**Alternatives considered**:
- *Single Vercel deployment with Python serverless functions*: Rejected because Vercel serverless has 10s (hobby) / 60s (pro) timeout limits. Sobol analysis with M3 can take minutes. Also, numpy in serverless cold starts is slow.
- *Single server (VPS) running both*: Rejected due to higher cost and operational overhead for <10 users. Free tiers of Vercel + Railway are sufficient.
- *Pyodide (Python in WebAssembly)*: Rejected because initial Pyodide load is ~10MB and startup is slow. numpy WASM performance is significantly worse than native.

**Rationale**: Vercel is optimized for Next.js with zero-config deployment. Railway supports long-running Docker containers with persistent connections, necessary for SSE streaming during computation. Both have free tiers adequate for <10 users.

### 2. SSE over WebSocket for Progress Streaming

**Decision**: Use Server-Sent Events (SSE) for streaming computation progress from backend to frontend.

**Alternatives considered**:
- *WebSocket*: More capable (bidirectional) but Vercel serverless doesn't support it. Would require additional infrastructure (e.g., Pusher, Ably). Overkill for one-way progress updates.
- *Polling*: Simpler but wastes bandwidth and adds latency. Progress would appear choppy.

**Rationale**: SSE is HTTP-based (works through proxies/CDNs), one-directional (sufficient for progress → browser), auto-reconnects natively, and FastAPI supports it via `StreamingResponse`. The browser connects directly to Railway for SSE streams, bypassing Vercel's serverless timeout.

### 3. Plotly.js for Charts

**Decision**: Use Plotly.js (`react-plotly.js`) for all interactive charts.

**Alternatives considered**:
- *Recharts*: Simpler API but lacks heatmap/contour support needed for mode shape visualization.
- *D3.js*: Maximum flexibility but too low-level for this scope. Would require significant custom code.
- *Chart.js*: Lightweight but no native heatmap support.
- *ECharts*: Good alternative with heatmap support, but Plotly has better scientific chart defaults and built-in export.

**Rationale**: Plotly.js natively supports all required chart types (line plots, heatmaps, grouped bar charts, subplots), includes interactive features (zoom, pan, hover, export to PNG/SVG), and has good React integration. The ~3MB bundle size is mitigated by dynamic importing on the analysis page only.

### 4. Zustand for State Management

**Decision**: Use Zustand with 3 separate stores (analysis, sensitivity, uncertainty).

**Alternatives considered**:
- *React Context + useReducer*: Sufficient but verbose. Zustand provides cleaner API with less boilerplate.
- *Redux Toolkit*: Too heavy for this application. No need for middleware, devtools, or normalized state.
- *Jotai/Recoil*: Atomic state good for forms but less natural for the action-oriented API calls here.

**Rationale**: Zustand is lightweight (~1KB), has simple API, works outside React components (useful for SSE callbacks), and supports middleware (persist, devtools) if needed later.

### 5. Next.js API Route as Proxy (Short Requests Only)

**Decision**: Short-lived API calls (Buckling single case, config validation) go through Next.js API route proxy. Long-running SSE streams connect directly from browser to Railway backend.

**Rationale**: The proxy hides the backend URL and handles CORS cleanly for simple requests. But Vercel serverless functions have timeout limits, so SSE streams must bypass Vercel entirely. CORS is configured on the FastAPI side to allow direct browser connections for SSE endpoints.

### 6. Thread Pool for CPU-Bound Computation

**Decision**: Use `concurrent.futures.ThreadPoolExecutor` in FastAPI to run numpy computations without blocking the event loop.

**Alternatives considered**:
- *ProcessPoolExecutor*: Better CPU isolation but higher overhead for process creation. Not needed for <10 concurrent users.
- *Celery/Redis*: Full task queue is overkill. No persistence or retry logic needed.

**Rationale**: numpy releases the GIL during array operations, so threading provides adequate concurrency. A thread pool with 2-4 workers handles the expected <10 concurrent users. In-memory job dict stores SSE generators keyed by job UUID.

### 7. KaTeX for Manual Equation Rendering

**Decision**: Use KaTeX for rendering mathematical equations in the HTML manual pages.

**Alternatives considered**:
- *MathJax*: More complete LaTeX support but 5x slower rendering and larger bundle.
- *Pre-rendered images*: Zero runtime cost but poor resolution on retina displays and not searchable.

**Rationale**: KaTeX renders 10-100x faster than MathJax, has a smaller bundle (~200KB vs ~1MB), and supports all equations used in the manuals (integrals, matrices, Greek letters, subscripts/superscripts).

### 8. Monorepo Structure

**Decision**: Keep frontend and backend in the same repository under `frontend/` and `backend/` directories.

**Rationale**: Single project, small team. Easier to keep API contracts in sync. Shared README and deployment documentation. No need for separate CI/CD pipelines at this scale.

## Risks / Trade-offs

**[Railway cold starts]** → First request after idle period (~15 min) may take 5-10 seconds as the container spins up. Mitigation: Frontend shows loading state with message "Connecting to computation server...". Consider Railway's health check to keep container warm if this becomes an issue.

**[M3 + Sobol computation time]** → Sobol with M3 solver and large N_base (>50) can take 10+ minutes. Mitigation: Show warning dialog before starting (same as desktop GUI). Recommend M2 for UQ. SSE progress shows real-time evaluation count. Add "Cancel" button that aborts the computation.

**[Plotly.js bundle size (~3MB)]** → Increases initial load of analysis page. Mitigation: Dynamic import with `next/dynamic({ ssr: false })`. Plotly only loads when user navigates to /analysis. Landing page and manual pages are unaffected.

**[SSE connection stability]** → Long-running SSE connections (minutes for Sobol) may drop due to network issues or proxy timeouts. Mitigation: Client implements auto-reconnect with exponential backoff. Backend supports resuming from last reported progress via job_id. Show "Reconnecting..." indicator.

**[Numerical precision across environments]** → numpy version differences between local and Railway could cause minor floating-point differences. Mitigation: Pin exact numpy version in backend requirements. Validation: compare desktop GUI output vs API output for test cases.

**[PDF-to-HTML manual conversion]** → Automated extraction may lose formatting, equations, or diagram quality. Mitigation: Manual review and hand-editing after initial extraction. Use KaTeX for equations and SVG for diagrams to ensure quality.
