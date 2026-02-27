## ADDED Requirements

### Requirement: Three-tab analysis interface
The system SHALL provide a tabbed interface with three tabs: "Buckling (Single Case)", "Sensitivity (OAT)", and "Uncertainty (Sobol UQ)". Tabs MUST be navigable without page reload.

#### Scenario: User switches between tabs
- **WHEN** user clicks on a tab header
- **THEN** the corresponding tab content is displayed immediately without page navigation, and input state is preserved across tab switches

### Requirement: Buckling input panel with all parameters
The system SHALL display an input panel with all beam parameters organized in collapsible sections: Geometry (L, b_root, b_tip, h_root, h_tip, w_f), Face Laminate (t_face_total, face_angles), Web Laminate (t_web_total, web_angles), Materials (Ef, Em, Gf, nuf, num, Vf), Boundary (Ktheta_root_per_m), and Solver (core selection M2/M3, PPW, nx_min). Each parameter MUST show its label, unit, and current value.

#### Scenario: All parameters displayed with defaults
- **WHEN** user opens the Buckling tab for the first time
- **THEN** all parameter fields are populated with default values matching the desktop GUI defaults (L=1.5, b_root=0.08, b_tip=0.04, h_root=0.025, h_tip=0.015, w_f=0.02, t_face_total=0.002, face_angles="0,45,-45,90", t_web_total=0.0015, web_angles="0,90", Ef=230e9, Em=3.5e9, Gf=90e9, nuf=0.2, num=0.35, Vf=0.6, Ktheta_root_per_m=1e9, PPW=60, nx_min=1801)

#### Scenario: Sections are collapsible
- **WHEN** user clicks a section header (e.g., "Geometry")
- **THEN** that section expands or collapses, showing or hiding its parameter inputs

### Requirement: Real-time input validation
The system SHALL validate each parameter field in real-time as the user types, showing visual feedback (green border for valid, red border with error message for invalid) using the same validation rules as the computation API.

#### Scenario: Valid input shows green feedback
- **WHEN** user enters a valid value (e.g., L = 2.0)
- **THEN** the input field shows a green border indicating validity

#### Scenario: Invalid input shows red feedback with message
- **WHEN** user enters an invalid value (e.g., Vf = 1.5)
- **THEN** the input field shows a red border and an error message "Vf must be between 0 and 1"

### Requirement: Parameter tooltips
The system SHALL display a tooltip on hover for each parameter label, explaining what the parameter represents, matching the PARAM_TOOLTIPS from the desktop GUI.

#### Scenario: Hover shows tooltip
- **WHEN** user hovers over a parameter label (e.g., "Beam Length (m)")
- **THEN** a tooltip appears with the description (e.g., "Beam length from root to tip (meters)")

### Requirement: Run buckling analysis
The system SHALL provide a "Run" button that sends current parameters to the backend API and displays results. While running, the button MUST be disabled and a loading indicator MUST be shown.

#### Scenario: Successful analysis run
- **WHEN** user clicks "Run" with valid parameters
- **THEN** a loading indicator appears, the API is called, and upon completion the results panel shows Pcr, dcr, mode parameters, load-deflection chart, and mode contour chart

#### Scenario: Run button disabled during analysis
- **WHEN** analysis is in progress
- **THEN** the Run button is disabled and shows "Running..." state

#### Scenario: Analysis error displayed
- **WHEN** the API returns an error
- **THEN** the error message is displayed in an alert/toast notification

### Requirement: Buckling result summary display
The system SHALL display analysis results including: Pcr (N and kN), dcr (mm), alpha* (1/m), beta* (1/m), lambda_x* (m), and the solver used (M2/M3).

#### Scenario: Results displayed after analysis
- **WHEN** buckling analysis completes successfully
- **THEN** a result summary card shows all values with appropriate units and formatting

### Requirement: Interactive load-deflection chart
The system SHALL render a Plotly.js chart showing three curves: Linear (dashed blue), Nonlinear (dotted orange), and Total (solid green), with a scatter marker at (dcr, Pcr). The chart MUST support zoom, pan, hover tooltips showing exact values, and export to PNG.

#### Scenario: Chart renders with three curves
- **WHEN** analysis results are available
- **THEN** the chart displays Linear, Nonlinear, and Total load-deflection curves with correct styles and a Pcr marker

#### Scenario: Chart is interactive
- **WHEN** user hovers over a data point on the chart
- **THEN** a tooltip shows the exact P and delta values at that point

### Requirement: Interactive mode contour chart
The system SHALL render a Plotly.js heatmap showing the normalized buckling mode shape (w/max|w|) over the beam domain (x vs y), using viridis colorscale with a colorbar.

#### Scenario: Contour chart renders mode shape
- **WHEN** analysis results include contour data
- **THEN** a heatmap displays the mode shape with x-axis (beam length) and y-axis (beam width), viridis colorscale, and colorbar labeled "w/max|w|"

### Requirement: Sensitivity parameter selection
The system SHALL display a list of all sweepable parameters with checkboxes, ±% input, and points input for each. Users MUST be able to select multiple parameters for simultaneous sweep.

#### Scenario: User configures sensitivity sweep
- **WHEN** user checks "Beam Length (m)", sets ±% to 15, and sets points to 7
- **THEN** that parameter is included in the sweep configuration with the specified range and resolution

### Requirement: Sensitivity analysis with progress
The system SHALL run sensitivity analysis via SSE, showing a progress bar with current/total evaluation count and a message indicating which parameter is being evaluated. Results MUST display as a grid of subplots (one per parameter).

#### Scenario: Progress updates during sensitivity run
- **WHEN** sensitivity analysis is running
- **THEN** a progress bar shows percentage complete and a message like "Evaluating L (5/35)"

#### Scenario: Results displayed as subplot grid
- **WHEN** sensitivity analysis completes
- **THEN** a grid of Plotly subplots shows Pcr vs parameter value for each swept parameter, with a dashed vertical line at the baseline value

### Requirement: Sobol parameter selection
The system SHALL display a list of all uncertain parameters with checkboxes, Low, and High bound inputs. Users MUST be able to set N_base, seed, and solver selection.

#### Scenario: User configures Sobol analysis
- **WHEN** user checks parameters and sets Low/High bounds, N_base=100, seed=1234
- **THEN** the configuration is ready for Sobol analysis execution

### Requirement: Update Sobol bounds from baseline
The system SHALL provide an "Update from Baseline" button that computes Low/High bounds as baseline ± uncertainty% for all parameters, using the current Buckling tab values as baseline.

#### Scenario: Bounds updated from baseline
- **WHEN** user sets uncertainty to 10% and clicks "Update from Baseline"
- **THEN** all parameter Low/High bounds are set to baseline * (1 ± 0.10)

### Requirement: Sobol analysis with progress
The system SHALL run Sobol analysis via SSE, showing progress with phase indicator (Matrix A, Matrix B, AB matrices). Results MUST display as a grouped bar chart sorted by ST descending.

#### Scenario: Progress updates during Sobol run
- **WHEN** Sobol analysis is running
- **THEN** a progress bar shows percentage complete with phase and evaluation count

#### Scenario: Results displayed as bar chart
- **WHEN** Sobol analysis completes
- **THEN** a Plotly grouped bar chart shows S1 (blue) and ST (coral) bars for each parameter, sorted by ST descending

### Requirement: JSON config save and load
The system SHALL allow users to save current input parameters as a JSON file (browser download) and load parameters from a JSON file (browser file upload). Loaded configs MUST be validated before applying.

#### Scenario: Save config downloads JSON
- **WHEN** user clicks "Save Config"
- **THEN** a JSON file containing all current parameter values and core selection is downloaded to the browser

#### Scenario: Load config populates inputs
- **WHEN** user clicks "Load Config" and selects a valid JSON file
- **THEN** all parameter inputs are populated with the values from the file

#### Scenario: Invalid config shows error
- **WHEN** user loads a JSON file with invalid or missing parameters
- **THEN** an error message indicates which parameters are invalid or missing

### Requirement: Export results
The system SHALL allow users to export analysis results as text (TXT file download) and export plots as PNG via Plotly's built-in download functionality.

#### Scenario: Export results as text
- **WHEN** user clicks "Export Results" after a buckling analysis
- **THEN** a TXT file is downloaded containing Pcr, dcr, mode parameters, and solver information

### Requirement: Responsive mobile layout
The system SHALL provide a usable layout on mobile devices (< 768px width). On mobile, input sections MUST be displayed as collapsible accordions stacked vertically, with results appearing below the inputs after scrolling.

#### Scenario: Mobile layout stacks content vertically
- **WHEN** user accesses the analysis page on a mobile device
- **THEN** input panel and result panel are stacked vertically (not side-by-side), input sections are collapsed by default, and results appear below

#### Scenario: Desktop layout uses split view
- **WHEN** user accesses the analysis page on a desktop (> 1024px)
- **THEN** input panel is on the left and result panel is on the right in a side-by-side layout

### Requirement: Geometry definition diagram
The system SHALL display a beam geometry diagram in the input panel showing the double-tapered box beam with labeled dimensions (L, b_root, b_tip, h_root, h_tip, w_f).

#### Scenario: Diagram displayed in geometry section
- **WHEN** user opens the Geometry input section
- **THEN** an SVG diagram of the beam cross-section and side view is displayed with dimension labels
