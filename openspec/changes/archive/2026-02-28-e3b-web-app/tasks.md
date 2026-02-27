## 1. Project Scaffolding

- [x] 1.1 Create `frontend/` directory with Next.js 15 (App Router, TypeScript, Tailwind CSS, `src/` directory)
- [x] 1.2 Install and configure shadcn/ui with required components (button, card, input, tabs, select, checkbox, accordion, progress, badge, alert, dialog, tooltip, skeleton)
- [x] 1.3 Create `backend/` directory with FastAPI project structure (app/main.py, routers/, services/, schemas/, engine/)
- [x] 1.4 Copy `buckling_analysis_M3a.py` into `backend/app/engine/` and verify import works
- [x] 1.5 Create backend `requirements.txt` (fastapi, uvicorn, numpy, pandas, pydantic)
- [x] 1.6 Create backend Dockerfile and railway.toml
- [x] 1.7 Set up frontend environment variables (`NEXT_PUBLIC_API_URL`) and Next.js config

## 2. Backend — Pydantic Schemas & Validation

- [x] 2.1 Create `schemas/buckling.py` with BucklingRequest (core + params) and BucklingResponse (Pcr, dcr, mode params, curves, contour)
- [x] 2.2 Create `schemas/sensitivity.py` with SensitivityRequest, SensJobResponse, SensProgressEvent, SensResultEvent
- [x] 2.3 Create `schemas/sobol.py` with SobolRequest, SobolJobResponse, SobolProgressEvent, SobolResultEvent
- [x] 2.4 Implement parameter validation rules (all ranges from spec) as Pydantic validators on request schemas

## 3. Backend — Buckling Endpoint

- [x] 3.1 Create `services/buckling_service.py` wrapping eval_m2_Pcr_with_mode() and eval_m3_Pcr_and_mode() + koiter_curves_from_mode()
- [x] 3.2 Implement mode contour grid generation (same logic as GUI plot_mode_contour, returns x/y/w_normalized arrays)
- [x] 3.3 Implement numpy array → list serialization for JSON response
- [x] 3.4 Create `routers/buckling.py` with POST `/api/buckling/run` using ThreadPoolExecutor for CPU-bound work
- [x] 3.5 Verify M2 and M3 results match desktop GUI output for test parameters

## 4. Backend — Sensitivity Endpoint with SSE

- [x] 4.1 Create `services/sensitivity_service.py` extracting OAT sweep logic from GUI sens_worker()
- [x] 4.2 Implement SSE async generator yielding progress events per evaluation
- [x] 4.3 Create in-memory job store (dict with UUID keys) for SSE stream management
- [x] 4.4 Create `routers/sensitivity.py` with POST `/api/sensitivity/run` (returns job_id) and GET `/api/sensitivity/stream/{job_id}` (SSE)

## 5. Backend — Sobol Endpoint with SSE

- [x] 5.1 Create `services/sobol_service.py` extracting Saltelli-lite Sobol logic from GUI sobol_worker()
- [x] 5.2 Implement SSE async generator yielding progress during A, B, AB matrix evaluations
- [x] 5.3 Create `routers/sobol.py` with POST `/api/sobol/run` and GET `/api/sobol/stream/{job_id}`

## 6. Backend — Config & Infrastructure

- [x] 6.1 Create `routers/config.py` with POST `/api/config/validate`
- [x] 6.2 Create `app/main.py` with FastAPI app, CORS middleware (Vercel + localhost), health check endpoint, and router registration
- [x] 6.3 Create `app/config.py` with Pydantic Settings for environment variables (ALLOWED_ORIGINS, etc.)
- [x] 6.4 Test all endpoints locally with curl/httpie

## 7. Frontend — Layout & Navigation

- [x] 7.1 Create root layout (`src/app/layout.tsx`) with metadata, font loading (Inter + JetBrains Mono), and global styles
- [x] 7.2 Create Navbar component with links (Home, Analysis, Manual) and responsive hamburger menu
- [x] 7.3 Create Footer component with version and credits
- [x] 7.4 Create `lib/utils.ts` with cn() helper and formatNumber() utility

## 8. Frontend — Core Utilities

- [x] 8.1 Create `lib/api.ts` — fetch wrapper for backend API calls (handles base URL, JSON, error responses)
- [x] 8.2 Create `lib/sse.ts` — EventSource wrapper with typed event handlers (onProgress, onResult, onDone, onError)
- [x] 8.3 Create `lib/validation.ts` — parameter validation rules matching backend (for real-time client-side feedback)
- [x] 8.4 Create `types/analysis.ts` — TypeScript types for BucklingParams, BucklingResults, CurveData, ContourData
- [x] 8.5 Create `types/sensitivity.ts` — types for SweepParam, SensitivityResults
- [x] 8.6 Create `types/uncertainty.ts` — types for UncertainParam, SobolResults
- [x] 8.7 Create `constants/defaults.ts` — default parameter values matching desktop GUI
- [x] 8.8 Create `constants/parameters.ts` — parameter metadata (key, label, unit, tooltip, validation rule, section)
- [x] 8.9 Create `constants/sobolDefaults.ts` — default Low/High bounds for Sobol parameters

## 9. Frontend — Zustand Stores

- [x] 9.1 Create `stores/analysisStore.ts` — params, core, results, isRunning, error, runAnalysis(), loadConfig(), exportConfig()
- [x] 9.2 Create `stores/sensitivityStore.ts` — sweepParams, core, progress, results, runSensitivity()
- [x] 9.3 Create `stores/uncertaintyStore.ts` — uncertainParams, nBase, seed, core, progress, results, runSobol(), updateFromBaseline()

## 10. Frontend — Buckling Tab Components

- [x] 10.1 Create `ParameterInput.tsx` — reusable input row (label, unit, tooltip, validation feedback, value binding)
- [x] 10.2 Create `GeometrySection.tsx` — accordion with geometry diagram SVG and 6 parameter inputs
- [x] 10.3 Create `LaminateSection.tsx` — accordion with face (2 inputs) and web (2 inputs) laminate sections
- [x] 10.4 Create `MaterialSection.tsx` — accordion with 6 material property inputs
- [x] 10.5 Create `BoundarySection.tsx` — accordion with root spring input
- [x] 10.6 Create `SolverSection.tsx` — accordion with M2/M3 radio selector and 2 solver parameter inputs
- [x] 10.7 Create `InputPanel.tsx` — assembles all sections with scrollable container
- [x] 10.8 Create `ResultSummary.tsx` — card displaying Pcr, dcr, alpha*, beta*, lambda_x* with units
- [x] 10.9 Create `DescriptionCard.tsx` — analysis method description text
- [x] 10.10 Create `LoadDeflectionChart.tsx` — Plotly line chart with 3 traces + Pcr marker (dynamic import)
- [x] 10.11 Create `ModeContourChart.tsx` — Plotly heatmap with viridis colorscale (dynamic import)
- [x] 10.12 Create `ResultPanel.tsx` — assembles description, summary, and charts
- [x] 10.13 Create `GeometryDiagram.tsx` — SVG inline diagram of double-tapered box beam with dimension labels

## 11. Frontend — Common Components

- [x] 11.1 Create `ProgressOverlay.tsx` — overlay with progress bar, percentage, message, and cancel button
- [x] 11.2 Create `ConfigManager.tsx` — save (browser download JSON) and load (file upload + validate) buttons
- [x] 11.3 Create `ExportMenu.tsx` — export results as TXT, plots via Plotly toolbar

## 12. Frontend — Analysis Page (Buckling Tab)

- [x] 12.1 Create `AnalysisTabs.tsx` — tab container with Buckling, Sensitivity, Uncertainty tabs
- [x] 12.2 Create `src/app/analysis/page.tsx` — analysis page assembling AnalysisTabs
- [x] 12.3 Wire InputPanel → analysisStore → API → ResultPanel end-to-end
- [x] 12.4 Implement responsive layout: split view on desktop, stacked on mobile
- [x] 12.5 Test Buckling tab: run M2 and M3, verify results and charts render correctly

## 13. Frontend — Sensitivity Tab

- [x] 13.1 Create `SensParamList.tsx` — scrollable list of parameters with checkbox, ±% input, points input
- [x] 13.2 Create `SensControls.tsx` — solver radio + Run button
- [x] 13.3 Create `SensitivityChart.tsx` — Plotly subplots grid (2 columns) with baseline dashed line
- [x] 13.4 Create `SensitivityTab.tsx` — assembles param list, controls, chart, and progress overlay
- [x] 13.5 Wire SSE connection: run → progress bar → result → chart render
- [x] 13.6 Implement responsive layout for sensitivity tab

## 14. Frontend — Uncertainty Tab

- [x] 14.1 Create `SobolParamList.tsx` — scrollable list with checkbox, Low, High inputs
- [x] 14.2 Create `SobolControls.tsx` — N_base input, seed input, uncertainty % input, "Update from Baseline" button, solver radio, Run button
- [x] 14.3 Create `SobolChart.tsx` — Plotly grouped bar chart (S1 blue, ST coral, sorted by ST)
- [x] 14.4 Create `UncertaintyTab.tsx` — assembles param list, controls, chart, and progress overlay
- [x] 14.5 Wire SSE connection: run → progress bar → result → chart render
- [x] 14.6 Implement "Update from Baseline" pulling values from analysisStore
- [x] 14.7 Implement responsive layout for uncertainty tab

## 15. Frontend — Next.js API Proxy

- [x] 15.1 Create `src/app/api/proxy/[...path]/route.ts` — proxy handler forwarding requests to FastAPI backend

## 16. Frontend — Manual Pages

- [x] 16.1 Read and extract content from User_Manual.pdf
- [x] 16.2 Read and extract content from Theory_Manual.pdf
- [x] 16.3 Install and configure KaTeX (or remark-math + rehype-katex) for equation rendering
- [x] 16.4 Create `ManualLayout.tsx` — sidebar TOC + content area layout
- [x] 16.5 Create `TableOfContents.tsx` — sticky sidebar with section links, collapsible on mobile
- [x] 16.6 Create `src/app/manual/page.tsx` — index page with links to both manuals
- [x] 16.7 Create `src/app/manual/user/page.tsx` — User Manual content with section headings and instructions
- [x] 16.8 Create `src/app/manual/theory/page.tsx` — Theory Manual content with KaTeX equations and diagrams

## 17. Frontend — Landing Page

- [x] 17.1 Create `HeroSection.tsx` — title, subtitle, description, "Start Analysis" CTA button
- [x] 17.2 Create `FeatureCards.tsx` — 3 cards (Buckling, Sensitivity, Uncertainty) with icons and descriptions
- [x] 17.3 Create `LogoBanner.tsx` — partner/institution logos
- [x] 17.4 Create `src/app/page.tsx` — landing page assembling hero, features, logos, footer

## 18. Frontend — Error Handling & Polish

- [x] 18.1 Create `src/app/error.tsx` — global error boundary with retry button
- [x] 18.2 Create `src/app/analysis/loading.tsx` — skeleton loading state for analysis page
- [x] 18.3 Add SEO metadata to all pages (title, description, og:tags)
- [x] 18.4 Configure Plotly dynamic imports (`next/dynamic` with `{ ssr: false }`) to optimize bundle

## 19. Deployment

- [x] 19.1 Deploy backend to Railway (Docker build, environment variables, health check)
- [x] 19.2 Deploy frontend to Vercel (connect repo, set NEXT_PUBLIC_API_URL, verify build)
- [x] 19.3 Verify CORS: frontend → backend API calls work in production
- [x] 19.4 Verify SSE: long-running sensitivity/sobol streams work end-to-end in production

## 20. End-to-End Verification

- [x] 20.1 Test Buckling tab: M2 and M3 results match desktop GUI (Pcr within 0.01%)
- [x] 20.2 Test Sensitivity tab: multi-parameter sweep with progress, chart renders correctly
- [x] 20.3 Test Sobol tab: full UQ analysis with progress, S1/ST chart correct
- [x] 20.4 Test JSON config save/load round-trip
- [x] 20.5 Test mobile layout: all 3 tabs usable on phone-width viewport
- [x] 20.6 Test manual pages: equations render, TOC navigation works
- [x] 20.7 Verify landing page loads and links work
