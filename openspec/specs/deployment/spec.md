## ADDED Requirements

### Requirement: Frontend deployment on Vercel
The system SHALL deploy the Next.js frontend application on Vercel with automatic builds from the repository. The deployment MUST serve the application at a public URL.

#### Scenario: Frontend accessible via public URL
- **WHEN** user navigates to the Vercel deployment URL
- **THEN** the landing page loads and all routes (/analysis, /manual/*) are accessible

### Requirement: Backend deployment on Railway
The system SHALL deploy the FastAPI backend as a Docker container on Railway. The deployment MUST support long-running SSE connections (minutes) without timeout.

#### Scenario: Backend API accessible
- **WHEN** frontend sends POST to the Railway backend URL
- **THEN** the backend processes the request and returns the response

#### Scenario: SSE stream stays connected for long computations
- **WHEN** a Sobol analysis runs for several minutes
- **THEN** the SSE connection remains open and continues streaming progress events until completion

### Requirement: Environment configuration
The system SHALL use environment variables for configuration: `NEXT_PUBLIC_API_URL` on Vercel pointing to the Railway backend URL, and `ALLOWED_ORIGINS` on Railway listing the Vercel frontend domain.

#### Scenario: Frontend connects to correct backend
- **WHEN** frontend makes an API call
- **THEN** it uses the `NEXT_PUBLIC_API_URL` environment variable to construct the request URL

### Requirement: Docker containerization for backend
The system SHALL provide a Dockerfile for the backend that installs Python 3.12, numpy, pandas, fastapi, uvicorn, and pydantic, and runs the FastAPI application.

#### Scenario: Docker build succeeds
- **WHEN** running `docker build` in the backend directory
- **THEN** the image builds successfully with all dependencies installed

#### Scenario: Container starts and serves API
- **WHEN** running the built Docker container
- **THEN** the FastAPI application starts on port 8000 and responds to health check requests

### Requirement: Landing page
The system SHALL serve a landing page at `/` with: project title ("E3B Buckling Analysis"), feature overview (Buckling, Sensitivity, Uncertainty), partner logos, and a "Start Analysis" button linking to `/analysis`.

#### Scenario: Landing page loads with features
- **WHEN** user navigates to the root URL
- **THEN** the landing page displays the project title, description, feature cards for each analysis mode, partner logos, and a CTA button to start analysis

### Requirement: Navigation bar
The system SHALL display a navigation bar on all pages with links to: Home (/), Analysis (/analysis), Manual (/manual). On mobile, the navigation MUST collapse into a hamburger menu.

#### Scenario: Desktop navigation visible
- **WHEN** user views any page on desktop
- **THEN** the navigation bar shows all links horizontally

#### Scenario: Mobile navigation as hamburger menu
- **WHEN** user views any page on mobile
- **THEN** the navigation collapses into a hamburger icon that expands to show links when tapped
