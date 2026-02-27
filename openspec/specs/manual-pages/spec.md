## ADDED Requirements

### Requirement: User Manual page
The system SHALL serve the User Manual content as an HTML page at `/manual/user`, covering: GUI overview, input parameter descriptions, how to run each analysis mode, how to interpret results, and config save/load instructions.

#### Scenario: User Manual loads with navigation
- **WHEN** user navigates to `/manual/user`
- **THEN** the User Manual page loads with a sidebar table of contents and the manual content rendered as structured HTML

### Requirement: Theory Manual page
The system SHALL serve the Theory Manual content as an HTML page at `/manual/theory`, covering: Ritz formulation, micromechanics (Halpin-Tsai), Classical Laminate Theory (ABD matrices), basis functions, eigenvalue problem, Koiter post-buckling theory, sensitivity analysis methodology, and Sobol indices.

#### Scenario: Theory Manual loads with equations
- **WHEN** user navigates to `/manual/theory`
- **THEN** the Theory Manual page loads with mathematical equations rendered via KaTeX, section headings, and sidebar navigation

### Requirement: Mathematical equation rendering
The system SHALL render all mathematical equations in the manuals using KaTeX. Equations MUST include integrals, matrices, Greek letters, subscripts, superscripts, and fractions as used in the original PDF manuals.

#### Scenario: Inline and block equations render correctly
- **WHEN** manual page contains inline math (e.g., $P_{cr}$) and display math (e.g., Ritz energy functional)
- **THEN** both inline and block equations render with proper mathematical formatting

### Requirement: Sidebar table of contents
The system SHALL display a sticky sidebar table of contents on each manual page. On desktop, the sidebar MUST remain visible while scrolling. On mobile, the sidebar MUST be collapsible.

#### Scenario: Desktop sidebar is sticky
- **WHEN** user scrolls through the manual on desktop
- **THEN** the sidebar TOC remains fixed in view, and clicking a TOC entry scrolls to the corresponding section

#### Scenario: Mobile sidebar is collapsible
- **WHEN** user views the manual on mobile
- **THEN** the TOC is hidden by default with a toggle button to show/hide it

### Requirement: Manual index page
The system SHALL serve a manual index page at `/manual` with links to both the User Manual and Theory Manual, including brief descriptions of each.

#### Scenario: Manual index links to both manuals
- **WHEN** user navigates to `/manual`
- **THEN** the page displays cards/links for "User Manual" and "Theory Manual" with brief descriptions
