## ADDED Requirements

### Requirement: Buckling single-case analysis endpoint
The system SHALL expose a POST endpoint at `/api/buckling/run` that accepts beam parameters and solver selection, executes the buckling analysis using the existing Python engine, and returns critical load (Pcr), critical deflection (dcr), mode parameters (alpha*, beta*, lambda_x*), load-deflection curve data, and mode contour grid data.

#### Scenario: M3 solver returns full results
- **WHEN** client sends POST to `/api/buckling/run` with `core: "m3"` and valid beam parameters
- **THEN** response contains Pcr, dcr, alpha_star, beta_star, lambda_x, curves (P, delta_linear, delta_nonlinear, delta_total arrays), and contour (x, y, w_normalized 2D grid data with nx and ny dimensions)

#### Scenario: M2 solver returns full results
- **WHEN** client sends POST to `/api/buckling/run` with `core: "m2"` and valid beam parameters
- **THEN** response contains Pcr, dcr, alpha_star, beta_star, lambda_x, curves, and contour data (same structure as M3)

#### Scenario: Invalid parameters rejected
- **WHEN** client sends POST with invalid parameters (e.g., L <= 0, Vf >= 1, non-numeric face_angles)
- **THEN** response status is 422 with an errors array listing each invalid field and the validation rule violated

#### Scenario: Numerical results match desktop GUI
- **WHEN** client sends identical parameters to API and desktop GUI
- **THEN** Pcr values match within floating-point tolerance (< 0.01% relative error)

### Requirement: Sensitivity analysis with SSE progress
The system SHALL expose a POST endpoint at `/api/sensitivity/run` that accepts baseline parameters and sweep configuration, returns a job_id immediately, and streams progress via SSE at `/api/sensitivity/stream/{job_id}`. Each selected parameter is swept ±percent around baseline with the specified number of points.

#### Scenario: Start sensitivity job and receive job_id
- **WHEN** client sends POST to `/api/sensitivity/run` with baseline_params, core, and sweep_params (array of {key, percent, points})
- **THEN** response contains job_id, total_evaluations count, and stream_url

#### Scenario: SSE streams progress events
- **WHEN** client connects to `/api/sensitivity/stream/{job_id}` via EventSource
- **THEN** server sends `progress` events with {current, total, param, message} for each evaluation, followed by a `result` event with {results: [{param, values, pcr_values}], baseline_pcr}, and finally a `done` event

#### Scenario: No parameters selected
- **WHEN** client sends POST with empty sweep_params array
- **THEN** response status is 422 with error message "At least one sweep parameter must be selected"

### Requirement: Sobol uncertainty analysis with SSE progress
The system SHALL expose a POST endpoint at `/api/sobol/run` that accepts baseline parameters, uncertain parameter bounds, N_base, and seed, returns a job_id immediately, and streams progress via SSE at `/api/sobol/stream/{job_id}`.

#### Scenario: Start Sobol job and receive job_id
- **WHEN** client sends POST to `/api/sobol/run` with baseline_params, core, uncertain_params (array of {key, low, high}), n_base, and seed
- **THEN** response contains job_id, total_evaluations (n_base * (2 + k) where k = number of uncertain params), and stream_url

#### Scenario: SSE streams Sobol progress
- **WHEN** client connects to `/api/sobol/stream/{job_id}` via EventSource
- **THEN** server sends `progress` events with {current, total, phase, message} during matrix A, B, and AB evaluations, followed by a `result` event with {names (sorted by ST descending), S1, ST arrays}, and a `done` event

#### Scenario: Invalid bounds rejected
- **WHEN** client sends POST with any uncertain_param where low >= high
- **THEN** response status is 422 with error message identifying the invalid parameter

### Requirement: Config validation endpoint
The system SHALL expose a POST endpoint at `/api/config/validate` that validates a set of beam parameters against all validation rules and returns success or a list of errors.

#### Scenario: Valid config
- **WHEN** client sends POST to `/api/config/validate` with all valid parameters
- **THEN** response contains `valid: true` and empty errors array

#### Scenario: Invalid config
- **WHEN** client sends POST with parameters violating validation rules
- **THEN** response contains `valid: false` and errors array with {field, message} entries for each violation

### Requirement: Parameter validation rules
The system SHALL validate all beam parameters according to these rules: L, b_root, b_tip, h_root, h_tip, w_f, t_face_total, t_web_total, Ef, Em, Gf MUST be > 0; nuf, num MUST be in range (-1, 0.5); Vf MUST be in range (0, 1); Ktheta_root_per_m MUST be >= 0; PPW, nx_min MUST be >= 10; face_angles, web_angles MUST be comma-separated valid numbers.

#### Scenario: All positive dimension parameters validated
- **WHEN** any of L, b_root, b_tip, h_root, h_tip, w_f, t_face_total, t_web_total, Ef, Em, Gf is <= 0
- **THEN** validation fails with error message "{field} must be greater than 0"

#### Scenario: Poisson ratio range validated
- **WHEN** nuf or num is outside range (-1, 0.5)
- **THEN** validation fails with error message "{field} must be between -1 and 0.5"

#### Scenario: Fiber volume fraction range validated
- **WHEN** Vf is outside range (0, 1)
- **THEN** validation fails with error message "Vf must be between 0 and 1"

### Requirement: CORS configuration
The system SHALL allow cross-origin requests from the Vercel frontend domain and localhost:3000 (development).

#### Scenario: Frontend domain allowed
- **WHEN** browser sends request with Origin header matching the configured Vercel domain
- **THEN** response includes proper CORS headers (Access-Control-Allow-Origin, etc.)

#### Scenario: Unknown origin blocked
- **WHEN** browser sends request with Origin header not in the allowed list
- **THEN** CORS headers are not included, and browser blocks the response

### Requirement: Health check endpoint
The system SHALL expose a GET endpoint at `/health` that returns 200 OK for deployment health monitoring.

#### Scenario: Health check passes
- **WHEN** client sends GET to `/health`
- **THEN** response status is 200 with body `{"status": "ok"}`
