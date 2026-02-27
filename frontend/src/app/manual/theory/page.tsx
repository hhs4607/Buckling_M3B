import { Card, CardContent } from "@/components/ui/card";
import { ManualLayout } from "@/components/manual/ManualLayout";
import { MathEquation } from "@/components/manual/MathEquation";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Theory Manual | E3B Buckling",
  description:
    "Comprehensive theoretical background for the E3B local face-buckling solver.",
};

const sections = [
  { id: "purpose-and-scope", title: "1. Purpose and Scope", level: 1 },
  { id: "solver-predicts", title: "1.1 What the Solver Predicts", level: 2 },
  { id: "run-modes", title: "1.2 Supported Run Modes", level: 2 },
  { id: "not-modeled", title: "1.3 What Is Not Modeled", level: 2 },
  { id: "geometry", title: "2. Geometry and Coordinates", level: 1 },
  { id: "coordinate-system", title: "2.1 Coordinate System", level: 2 },
  { id: "primary-geometry", title: "2.2 Primary Geometry", level: 2 },
  { id: "derived-fields", title: "2.3 Derived Fields", level: 2 },
  { id: "material-modeling", title: "3. Material Modeling", level: 1 },
  { id: "constituent-inputs", title: "3.1 Constituent Inputs", level: 2 },
  { id: "micromechanics", title: "3.2 Micromechanics Outputs", level: 2 },
  { id: "q-matrix", title: "3.3 Reduced Stiffness Q and Q-bar", level: 2 },
  { id: "clt", title: "4. Classical Lamination Theory", level: 1 },
  { id: "layup", title: "4.1 Layup", level: 2 },
  { id: "abd", title: "4.2 ABD Computation", level: 2 },
  { id: "abd-usage", title: "4.3 How Laminate Matrices Are Used", level: 2 },
  { id: "load-mapping", title: "5. Load Mapping and Section Properties", level: 1 },
  { id: "thin-wall", title: "5.1 Thin-Wall Mapping", level: 2 },
  { id: "section-inertia", title: "5.2 Section Moment of Inertia", level: 2 },
  { id: "alpha0", title: "5.3 alpha0(x)", level: 2 },
  { id: "ei-total", title: "5.4 EI_total(x)", level: 2 },
  { id: "buckling", title: "6. Buckling Formulation (Rayleigh-Ritz)", level: 1 },
  { id: "displacement-field", title: "6.1 Assumed Displacement Field", level: 2 },
  { id: "longitudinal-basis", title: "6.2 Longitudinal Basis Functions", level: 2 },
  { id: "energy-structure", title: "6.3 Energy Structure", level: 2 },
  { id: "eigenproblem", title: "6.4 Generalized Eigenvalue Problem", level: 2 },
  { id: "search-strategy", title: "6.5 Search Strategy", level: 2 },
  { id: "edge-rotation", title: "7. Edge-Rotation Modeling", level: 1 },
  { id: "web-restraint", title: "7.1 Web Rotational Restraint", level: 2 },
  { id: "root-spring", title: "7.2 Root Rotational Spring (M3)", level: 2 },
  { id: "flanges-rotation", title: "7.3 Flanges", level: 2 },
  { id: "post-buckling", title: "8. Post-Buckling (Corrected Koiter)", level: 1 },
  { id: "koiter-overview", title: "8.1 Overview", level: 2 },
  { id: "amplitude-den2", title: "8.2 Amplitude Equation and Den2", level: 2 },
  { id: "quartic-k4", title: "8.3 Quartic Coefficient k4", level: 2 },
  { id: "deflection-decomposition", title: "8.4 Deflection Decomposition", level: 2 },
  { id: "numerical", title: "9. Numerical Implementation", level: 1 },
  { id: "discretization", title: "9.1 Discretization", level: 2 },
  { id: "integration", title: "9.2 Integration", level: 2 },
  { id: "mode-normalization", title: "9.3 Mode Normalization", level: 2 },
  { id: "stability-guards", title: "9.4 Stability Guards", level: 2 },
  { id: "solver-modes", title: "10. Solver Modes and Outputs", level: 1 },
  { id: "full-mode", title: "10.1 FULL Mode", level: 2 },
  { id: "sens-mode", title: "10.2 SENS Mode", level: 2 },
  { id: "sobol-mode", title: "10.3 SOBOL Mode", level: 2 },
  { id: "verification", title: "11. Verification and Examples", level: 1 },
  { id: "fem-benchmark", title: "11.1 FEM Benchmark", level: 2 },
  { id: "example-cases", title: "11.2 Example Cases", level: 2 },
  { id: "limitations", title: "12. Limitations and Recommended Use", level: 1 },
  { id: "appendix-c", title: "Appendix C: Key Equations", level: 1 },
];

export default function TheoryManualPage() {
  return (
    <ManualLayout title="Theory Manual" sections={sections}>
      <div className="space-y-10">
        {/* ================================================================
            Section 1 – Purpose and Scope
            ================================================================ */}
        <section>
          <h2 id="purpose-and-scope" className="text-xl font-semibold mb-4">
            1. Purpose and Scope
          </h2>

          {/* 1.1 */}
          <div className="mb-6">
            <h3 id="solver-predicts" className="text-lg font-medium mb-2">
              1.1 What the Solver Predicts
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The solver predicts the critical local face-buckling load (
                  <MathEquation math="P_{cr}" />) of a thin-walled, laminated,
                  double-tapered box beam subjected to a transverse tip load in a
                  cantilever configuration. Local buckling is represented by a
                  single sinusoidal half-wave across the interior face panel and a
                  Ritz basis along the span.
                </p>
                <p className="mt-2">
                  In <strong>FULL</strong> mode the solver also generates a
                  Koiter-type post-buckling curve and a normalized mode-shape
                  contour.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 1.2 */}
          <div className="mb-6">
            <h3 id="run-modes" className="text-lg font-medium mb-2">
              1.2 Supported Run Modes
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <ul className="list-disc list-inside space-y-1">
                  <li>
                    <strong>FULL</strong> — Single-case analysis producing{" "}
                    <MathEquation math="P_{cr}" />, deflection curves, and mode
                    contour.
                  </li>
                  <li>
                    <strong>SENS</strong> — One-at-a-time (OAT) parameter sweep
                    around a baseline.
                  </li>
                  <li>
                    <strong>SOBOL</strong> — Saltelli-lite uncertainty
                    quantification producing first-order (
                    <MathEquation math="S_i" />) and total-order (
                    <MathEquation math="S_{Ti}" />) indices.
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* 1.3 */}
          <div className="mb-6">
            <h3 id="not-modeled" className="text-lg font-medium mb-2">
              1.3 What Is Not Modeled
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>The following phenomena are explicitly excluded from the solver:</p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>Global beam instabilities (lateral-torsional buckling)</li>
                  <li>Progressive damage and material nonlinearity</li>
                  <li>
                    Geometric nonlinearity beyond the simplified Koiter formulation
                  </li>
                  <li>Imperfection sensitivity</li>
                  <li>Local web or flange buckling</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 2 – Geometry and Coordinate Definitions
            ================================================================ */}
        <section>
          <h2 id="geometry" className="text-xl font-semibold mb-4">
            2. Geometry and Coordinate Definitions
          </h2>

          {/* 2.1 */}
          <div className="mb-6">
            <h3 id="coordinate-system" className="text-lg font-medium mb-2">
              2.1 Coordinate System
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The cantilever beam is oriented with the spanwise coordinate{" "}
                  <MathEquation math="x" /> running from the clamped root (
                  <MathEquation math="x = 0" />) to the free tip (
                  <MathEquation math="x = L" />
                  ). The width coordinate <MathEquation math="y" /> spans the face
                  panel between the webs. The thickness coordinate{" "}
                  <MathEquation math="z" /> is normal to the laminate mid-surface.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 2.2 */}
          <div className="mb-6">
            <h3 id="primary-geometry" className="text-lg font-medium mb-2">
              2.2 Primary Geometry
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>The following dimensional parameters define the box-beam geometry:</p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    <MathEquation math="L" /> — Beam span length [m]
                  </li>
                  <li>
                    <MathEquation math="b_{\text{root}}" />,{" "}
                    <MathEquation math="b_{\text{tip}}" /> — Face-panel width at
                    root and tip [m]
                  </li>
                  <li>
                    <MathEquation math="h_{\text{root}}" />,{" "}
                    <MathEquation math="h_{\text{tip}}" /> — Core height at root
                    and tip [m]
                  </li>
                  <li>
                    <MathEquation math="w_f" /> — Flange width [m]
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* 2.3 */}
          <div className="mb-6">
            <h3 id="derived-fields" className="text-lg font-medium mb-2">
              2.3 Derived Fields
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Because the beam is double-tapered, width and height vary
                  linearly along the span:
                </p>
                <MathEquation
                  math="b(x) = b_{\text{root}} + \bigl(b_{\text{tip}} - b_{\text{root}}\bigr)\,\frac{x}{L}"
                  display
                />
                <MathEquation
                  math="h(x) = h_{\text{root}} + \bigl(h_{\text{tip}} - h_{\text{root}}\bigr)\,\frac{x}{L}"
                  display
                />
                <p>
                  The outer section height includes the face laminates:
                </p>
                <MathEquation
                  math="H(x) = h(x) + 2\,t_{\text{face,total}}"
                  display
                />
                <p>
                  The transverse wave number used in the sinusoidal buckling mode
                  is:
                </p>
                <MathEquation math="k_y(x) = \frac{\pi}{b(x)}" display />
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 3 – Material Modeling
            ================================================================ */}
        <section>
          <h2 id="material-modeling" className="text-xl font-semibold mb-4">
            3. Material Modeling
          </h2>

          {/* 3.1 */}
          <div className="mb-6">
            <h3 id="constituent-inputs" className="text-lg font-medium mb-2">
              3.1 Constituent Inputs
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The micromechanics module requires the following constituent
                  properties:
                </p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    <MathEquation math="E_f" /> — Fiber Young&apos;s modulus [Pa]
                  </li>
                  <li>
                    <MathEquation math="G_f" /> — Fiber shear modulus [Pa]
                  </li>
                  <li>
                    <MathEquation math="\nu_f" /> — Fiber Poisson&apos;s ratio [–]
                  </li>
                  <li>
                    <MathEquation math="E_m" /> — Matrix Young&apos;s modulus [Pa]
                  </li>
                  <li>
                    <MathEquation math="\nu_m" /> — Matrix Poisson&apos;s ratio [–]
                  </li>
                  <li>
                    <MathEquation math="V_f" /> — Fiber volume fraction [–]
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* 3.2 */}
          <div className="mb-6">
            <h3 id="micromechanics" className="text-lg font-medium mb-2">
              3.2 Micromechanics Outputs
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Effective unidirectional ply constants are computed from the
                  constituent properties:
                </p>
                <ul className="list-disc list-inside space-y-2 mt-2">
                  <li>
                    <MathEquation math="E_1" /> — Longitudinal modulus (rule of
                    mixtures):
                    <MathEquation
                      math="E_1 = E_f\,V_f + E_m\,(1 - V_f)"
                      display
                    />
                  </li>
                  <li>
                    <MathEquation math="E_2" /> and{" "}
                    <MathEquation math="G_{12}" /> — Transverse modulus and
                    in-plane shear modulus computed via the Halpin-Tsai
                    semi-empirical equations.
                  </li>
                  <li>
                    <MathEquation math="\nu_{12}" /> — Major Poisson&apos;s ratio
                    (linear mixture):
                    <MathEquation
                      math="\nu_{12} = \nu_f\,V_f + \nu_m\,(1 - V_f)"
                      display
                    />
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* 3.3 */}
          <div className="mb-6">
            <h3 id="q-matrix" className="text-lg font-medium mb-2">
              3.3 Reduced Stiffness Q and Transformed Q-bar
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The plane-stress reduced stiffness matrix{" "}
                  <MathEquation math="\mathbf{Q}" /> is formed from the ply
                  constants{" "}
                  <MathEquation math="(E_1,\,E_2,\,G_{12},\,\nu_{12})" />:
                </p>
                <MathEquation
                  math="\mathbf{Q} = \begin{bmatrix} Q_{11} & Q_{12} & 0 \\ Q_{12} & Q_{22} & 0 \\ 0 & 0 & Q_{66} \end{bmatrix}"
                  display
                />
                <p>where</p>
                <MathEquation
                  math="Q_{11} = \frac{E_1}{1 - \nu_{12}\nu_{21}},\quad Q_{22} = \frac{E_2}{1 - \nu_{12}\nu_{21}},\quad Q_{12} = \nu_{12}\,Q_{22},\quad Q_{66} = G_{12}"
                  display
                />
                <p>
                  For each ply with orientation angle{" "}
                  <MathEquation math="\theta" />, the transformed stiffness matrix{" "}
                  <MathEquation math="\bar{\mathbf{Q}}" /> is computed via the
                  standard tensor transformation.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 4 – Classical Lamination Theory
            ================================================================ */}
        <section>
          <h2 id="clt" className="text-xl font-semibold mb-4">
            4. Classical Lamination Theory
          </h2>

          {/* 4.1 */}
          <div className="mb-6">
            <h3 id="layup" className="text-lg font-medium mb-2">
              4.1 Layup
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Two laminates are defined independently: the <strong>face</strong>{" "}
                  laminate (top and bottom panels) and the <strong>web</strong>{" "}
                  laminate (left and right webs). Each laminate is specified by a
                  sequence of ply angles and a total laminate thickness.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 4.2 */}
          <div className="mb-6">
            <h3 id="abd" className="text-lg font-medium mb-2">
              4.2 ABD Computation
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The transformed stiffness{" "}
                  <MathEquation math="\bar{\mathbf{Q}}" /> is integrated through
                  the laminate thickness <MathEquation math="z" /> to form the ABD
                  stiffness matrices:
                </p>
                <MathEquation
                  math="A_{ij} = \sum_{k=1}^{N} \bar{Q}_{ij}^{(k)}\,(z_k - z_{k-1})"
                  display
                />
                <MathEquation
                  math="B_{ij} = \frac{1}{2}\sum_{k=1}^{N} \bar{Q}_{ij}^{(k)}\,(z_k^2 - z_{k-1}^2)"
                  display
                />
                <MathEquation
                  math="D_{ij} = \frac{1}{3}\sum_{k=1}^{N} \bar{Q}_{ij}^{(k)}\,(z_k^3 - z_{k-1}^3)"
                  display
                />
                <p>
                  Here <MathEquation math="N" /> is the total number of plies and{" "}
                  <MathEquation math="z_k" /> are the ply boundary coordinates
                  measured from the laminate mid-plane.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 4.3 */}
          <div className="mb-6">
            <h3 id="abd-usage" className="text-lg font-medium mb-2">
              4.3 How Laminate Matrices Are Used
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <ul className="list-disc list-inside space-y-2">
                  <li>
                    <MathEquation math="\mathbf{D}_{\text{face}}" /> — Bending
                    energy in the Rayleigh-Ritz formulation (dominant stiffness
                    controlling <MathEquation math="P_{cr}" />).
                  </li>
                  <li>
                    <MathEquation math="\mathbf{A}_{\text{face}}" /> — Membrane
                    stiffness used in the Koiter quartic coefficient.
                  </li>
                  <li>
                    <MathEquation math="\mathbf{D}_{\text{web}}" /> — Web bending
                    stiffness providing edge rotational restraint to the face
                    panel.
                  </li>
                  <li>
                    <MathEquation math="\mathbf{A}_{\text{web}}" /> — Contributes
                    to the overall section <MathEquation math="EI" />.
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 5 – Load Mapping and Section Properties
            ================================================================ */}
        <section>
          <h2 id="load-mapping" className="text-xl font-semibold mb-4">
            5. Load Mapping and Section Properties
          </h2>

          {/* 5.1 */}
          <div className="mb-6">
            <h3 id="thin-wall" className="text-lg font-medium mb-2">
              5.1 Thin-Wall Mapping
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  For a cantilever with tip load <MathEquation math="P" />, the
                  bending moment distribution is:
                </p>
                <MathEquation math="M(x) = P\,(L - x)" display />
                <p>
                  This moment is mapped to a face-panel axial stress resultant via
                  the section moment of inertia.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 5.2 */}
          <div className="mb-6">
            <h3 id="section-inertia" className="text-lg font-medium mb-2">
              5.2 Section Moment of Inertia{" "}
              <MathEquation math="I_{\text{total}}(x)" />
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The total section moment of inertia{" "}
                  <MathEquation math="I_{\text{total}}(x)" /> is assembled from
                  three contributions:
                </p>
                <ol className="list-decimal list-inside space-y-1 mt-2">
                  <li>
                    <strong>Face panels</strong> at{" "}
                    <MathEquation math="\pm H(x)/2" /> (parallel-axis theorem).
                  </li>
                  <li>
                    <strong>Webs</strong> modeled as rectangular sections
                    contributing their own second moment of area.
                  </li>
                  <li>
                    <strong>Flanges</strong> (bonded build-ups) contributing
                    additional area moment.
                  </li>
                </ol>
              </CardContent>
            </Card>
          </div>

          {/* 5.3 */}
          <div className="mb-6">
            <h3 id="alpha0" className="text-lg font-medium mb-2">
              5.3 <MathEquation math="\alpha_0(x)" />
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The function <MathEquation math="\alpha_0(x)" /> is the
                  spanwise weighting that scales the geometric/work term in the
                  energy functional. It is proportional to{" "}
                  <MathEquation math="(L - x)" /> and inversely related to{" "}
                  <MathEquation math="I_{\text{total}}(x)" />:
                </p>
                <MathEquation
                  math="\alpha_0(x) \;\propto\; \frac{L - x}{I_{\text{total}}(x)}"
                  display
                />
                <p>
                  This means <MathEquation math="\alpha_0" /> is large near the
                  root (where the bending moment is highest) and diminishes toward
                  the tip.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 5.4 */}
          <div className="mb-6">
            <h3 id="ei-total" className="text-lg font-medium mb-2">
              5.4 <MathEquation math="EI_{\text{total}}(x)" />
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The equivalent bending stiffness{" "}
                  <MathEquation math="EI_{\text{total}}(x)" /> is used for
                  computing linear tip deflection. It is assembled from the axial
                  stiffness <MathEquation math="A_{11}" /> of each
                  sub-component: face panels, webs, and flanges.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 6 – Buckling Formulation (Rayleigh-Ritz)
            ================================================================ */}
        <section>
          <h2 id="buckling" className="text-xl font-semibold mb-4">
            6. Buckling Formulation (Rayleigh-Ritz)
          </h2>

          {/* 6.1 */}
          <div className="mb-6">
            <h3 id="displacement-field" className="text-lg font-medium mb-2">
              6.1 Assumed Displacement Field
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The out-of-plane displacement of the face panel is assumed to be
                  separable in <MathEquation math="x" /> and{" "}
                  <MathEquation math="y" />:
                </p>
                <MathEquation
                  math="w(x,y) = F(x) \cdot \phi(y)"
                  display
                />
                <p>
                  where the transverse mode shape is a single sinusoidal
                  half-wave:
                </p>
                <MathEquation
                  math="\phi(y) = \sin\!\left(\frac{\pi\,y}{b(x)}\right)"
                  display
                />
                <p>
                  This choice automatically satisfies simply-supported conditions
                  at the web-face junctions.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 6.2 */}
          <div className="mb-6">
            <h3 id="longitudinal-basis" className="text-lg font-medium mb-2">
              6.2 Longitudinal Basis Functions
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>Two model variants are available:</p>
                <div className="mt-3">
                  <p>
                    <strong>M2 (one-term):</strong> A single basis function
                    captures the spanwise buckling shape:
                  </p>
                  <MathEquation
                    math="F_2(x) = e^{-\alpha x}\,x\,\sin(\beta x)"
                    display
                  />
                </div>
                <div className="mt-3">
                  <p>
                    <strong>M3 (two-term):</strong> A richer approximation with
                    two basis functions:
                  </p>
                  <MathEquation
                    math="F(x) = c_1\,F_1(x) + c_2\,F_2(x)"
                    display
                  />
                  <p>where</p>
                  <MathEquation
                    math="F_1(x) = e^{-\alpha x}\,\sin(\beta x)"
                    display
                  />
                  <MathEquation
                    math="F_2(x) = e^{-\alpha x}\,x\,\sin(\beta x)"
                    display
                  />
                  <p>
                    The coefficients <MathEquation math="c_1" /> and{" "}
                    <MathEquation math="c_2" /> are determined by the eigenvalue
                    problem.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 6.3 */}
          <div className="mb-6">
            <h3 id="energy-structure" className="text-lg font-medium mb-2">
              6.3 Energy Structure
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The total potential energy is minimized with respect to the
                  unknown coefficients. The energy contributions are:
                </p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    <strong>Bending energy</strong> — from{" "}
                    <MathEquation math="\mathbf{D}_{\text{face}}" /> components (
                    <MathEquation math="D_{11}" />,{" "}
                    <MathEquation math="D_{12}" />,{" "}
                    <MathEquation math="D_{22}" />,{" "}
                    <MathEquation math="D_{66}" />
                    ).
                  </li>
                  <li>
                    <strong>Coupling terms</strong> — mixed derivatives of{" "}
                    <MathEquation math="F(x)" /> and{" "}
                    <MathEquation math="\phi(y)" />.
                  </li>
                  <li>
                    <strong>Shear terms</strong> — from{" "}
                    <MathEquation math="D_{66}" /> twist energy.
                  </li>
                  <li>
                    <strong>Edge rotation penalty</strong> — rotational spring
                    energy at web-face junctions.
                  </li>
                </ul>
                <p className="mt-2">
                  These assemble into the stiffness matrix{" "}
                  <MathEquation math="\mathbf{N}" />. The geometric/work term,
                  built from <MathEquation math="\alpha_0(x)" /> and{" "}
                  <MathEquation math="F'(x)" />, forms the matrix{" "}
                  <MathEquation math="\mathbf{D}" />.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 6.4 */}
          <div className="mb-6">
            <h3 id="eigenproblem" className="text-lg font-medium mb-2">
              6.4 Generalized Eigenvalue Problem
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The stationary condition on total potential energy yields the
                  generalized eigenvalue problem:
                </p>
                <MathEquation
                  math="\mathbf{N}\,\mathbf{c} = P\,\mathbf{D}\,\mathbf{c}"
                  display
                />
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    <strong>M2:</strong> scalar problem (1 unknown). The critical
                    load is simply{" "}
                    <MathEquation math="P_{cr} = N_{22}/D_{22}" />.
                  </li>
                  <li>
                    <strong>M3:</strong> 2-component eigenproblem. The critical
                    load is the minimum eigenvalue of the{" "}
                    <MathEquation math="2 \times 2" /> system.
                  </li>
                </ul>
                <p className="mt-2">
                  In both cases, <MathEquation math="P_{cr}" /> is the minimum
                  eigenvalue found over the scanned{" "}
                  <MathEquation math="(\alpha,\,\beta)" /> parameter space.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 6.5 */}
          <div className="mb-6">
            <h3 id="search-strategy" className="text-lg font-medium mb-2">
              6.5 Search Strategy
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The minimum of{" "}
                  <MathEquation math="P_{cr}(\alpha,\,\beta)" /> is located by a
                  two-stage process:
                </p>
                <ol className="list-decimal list-inside space-y-1 mt-2">
                  <li>
                    <strong>Coarse scan</strong> — Evaluate{" "}
                    <MathEquation math="P_{cr}" /> on a regular grid in{" "}
                    <MathEquation math="(\alpha,\,\beta)" /> space.
                  </li>
                  <li>
                    <strong>Local refinement</strong> — Refine around the coarse
                    minimum to obtain a robust global minimum.
                  </li>
                </ol>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 7 – Edge-Rotation Modeling
            ================================================================ */}
        <section>
          <h2 id="edge-rotation" className="text-xl font-semibold mb-4">
            7. Edge-Rotation Modeling
          </h2>

          {/* 7.1 */}
          <div className="mb-6">
            <h3 id="web-restraint" className="text-lg font-medium mb-2">
              7.1 Web Rotational Restraint
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Each web provides rotational restraint to the face panel along
                  the web-face junction. The combined rotational stiffness from
                  both webs (per unit spanwise length) is:
                </p>
                <MathEquation
                  math="k_{\theta,\text{web-pair}}(x) = 2 \cdot \frac{4\,D_{22}^{\text{web}}}{h(x)}"
                  display
                />
                <p>
                  This stiffness enters the buckling energy as an edge penalty
                  term, stiffening the face panel against rotation at the
                  web-face boundary.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 7.2 */}
          <div className="mb-6">
            <h3 id="root-spring" className="text-lg font-medium mb-2">
              7.2 Root Rotational Spring (M3 only)
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  In the M3 model an additional rotational spring{" "}
                  <MathEquation math="K_{\theta,\text{root}}" /> [N&middot;m/m]
                  is applied at the clamped root. This spring enters the stiffness
                  matrix element <MathEquation math="N_{11}" /> as an energy
                  penalty proportional to{" "}
                  <MathEquation math="\beta^2" /> and the root half-width.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 7.3 */}
          <div className="mb-6">
            <h3 id="flanges-rotation" className="text-lg font-medium mb-2">
              7.3 Flanges
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Flanges do not contribute to edge rotational restraint. They are
                  assumed free in <MathEquation math="y" />-rotation. Their role
                  is limited to augmenting the section bending stiffness{" "}
                  <MathEquation math="EI_{\text{total}}" /> and moment of inertia{" "}
                  <MathEquation math="I_{\text{total}}" />.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 8 – Post-Buckling (Corrected Koiter)
            ================================================================ */}
        <section>
          <h2 id="post-buckling" className="text-xl font-semibold mb-4">
            8. Post-Buckling (Corrected Koiter)
          </h2>

          {/* 8.1 */}
          <div className="mb-6">
            <h3 id="koiter-overview" className="text-lg font-medium mb-2">
              8.1 Overview
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Beyond the bifurcation point, a simplified Koiter-type model
                  describes the post-buckling amplitude growth. This captures the
                  initial portion of the nonlinear load-deflection response
                  without a full nonlinear analysis.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 8.2 */}
          <div className="mb-6">
            <h3 id="amplitude-den2" className="text-lg font-medium mb-2">
              8.2 Amplitude Equation and Den2
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The post-buckling amplitude{" "}
                  <MathEquation math="a(P)" /> is computed from the load
                  increment beyond <MathEquation math="P_{cr}" />:
                </p>
                <MathEquation
                  math="a(P) = \sqrt{\frac{(P - P_{cr})\,\text{Den2}}{k_4}}"
                  display
                />
                <p>
                  The denominator integral <strong>Den2</strong> is defined as:
                </p>
                <MathEquation
                  math="\text{Den2} = \int_0^L \alpha_0(x)\,\bigl[F'(x)\bigr]^2\,\frac{b(x)}{2}\,dx"
                  display
                />
                <p>
                  Den2 effectively measures how much geometric work the critical
                  mode shape produces.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 8.3 */}
          <div className="mb-6">
            <h3 id="quartic-k4" className="text-lg font-medium mb-2">
              8.3 Quartic Coefficient <MathEquation math="k_4" />
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The quartic coefficient <MathEquation math="k_4" /> controls
                  the post-buckling stiffness. It is assembled from the face
                  membrane stiffness terms:
                </p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    <MathEquation math="A_{11}" />,{" "}
                    <MathEquation math="A_{12}" />,{" "}
                    <MathEquation math="A_{22}" />,{" "}
                    <MathEquation math="A_{66}" /> of the face laminate.
                  </li>
                </ul>
                <p className="mt-2">
                  These are combined with exact width integrals of the sinusoidal
                  mode shape <MathEquation math="\phi(y)" /> (products of{" "}
                  <MathEquation math="\sin" /> and <MathEquation math="\cos" />{" "}
                  terms).
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 8.4 */}
          <div className="mb-6">
            <h3 id="deflection-decomposition" className="text-lg font-medium mb-2">
              8.4 Deflection Decomposition
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The total tip deflection is decomposed into a linear
                  (pre-buckling) part and a local post-buckling contribution:
                </p>
                <MathEquation
                  math="\delta_{\text{lin}} = P \cdot f_b"
                  display
                />
                <p>
                  where <MathEquation math="f_b" /> is the linear bending
                  flexibility. The local post-buckling deflection is:
                </p>
                <MathEquation
                  math="\delta_{\text{loc}} = L \cdot \theta_{\text{fac}} \cdot a(P)"
                  display
                />
                <p>The total deflection is simply:</p>
                <MathEquation
                  math="\delta_{\text{tot}} = \delta_{\text{lin}} + \delta_{\text{loc}}"
                  display
                />
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 9 – Numerical Implementation
            ================================================================ */}
        <section>
          <h2 id="numerical" className="text-xl font-semibold mb-4">
            9. Numerical Implementation
          </h2>

          {/* 9.1 */}
          <div className="mb-6">
            <h3 id="discretization" className="text-lg font-medium mb-2">
              9.1 Discretization
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The number of spanwise grid points is chosen as:
                </p>
                <MathEquation
                  math="n_x = \max\!\bigl(n_{x,\min},\;\lceil (L / \lambda_{\text{guess}}) \cdot \text{PPW} \rceil\bigr)"
                  display
                />
                <p>
                  where <MathEquation math="\text{PPW}" /> is the user-specified
                  points per wavelength and{" "}
                  <MathEquation math="\lambda_{\text{guess}}" /> is an initial
                  wavelength estimate.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 9.2 */}
          <div className="mb-6">
            <h3 id="integration" className="text-lg font-medium mb-2">
              9.2 Integration
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  All spanwise integrals (energy terms, Den2, etc.) are evaluated
                  numerically using the <strong>trapezoidal rule</strong> over the
                  uniform <MathEquation math="x" />-grid.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 9.3 */}
          <div className="mb-6">
            <h3 id="mode-normalization" className="text-lg font-medium mb-2">
              9.3 Mode Normalization
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The longitudinal derivative{" "}
                  <MathEquation math="F'(x)" /> is normalized by its{" "}
                  <MathEquation math="L^2" /> norm to ensure consistent energy
                  scaling:
                </p>
                <MathEquation
                  math="\|F'\|_{L^2} = \sqrt{\int_0^L \bigl[F'(x)\bigr]^2\,dx}"
                  display
                />
              </CardContent>
            </Card>
          </div>

          {/* 9.4 */}
          <div className="mb-6">
            <h3 id="stability-guards" className="text-lg font-medium mb-2">
              9.4 Stability Guards
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  To prevent division-by-zero in degenerate configurations, small
                  positive floors (
                  <MathEquation math="\sim 10^{-18}" />) are applied to all
                  denominators throughout the solver.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 10 – Solver Modes and Outputs
            ================================================================ */}
        <section>
          <h2 id="solver-modes" className="text-xl font-semibold mb-4">
            10. Solver Modes and Outputs
          </h2>

          {/* 10.1 */}
          <div className="mb-6">
            <h3 id="full-mode" className="text-lg font-medium mb-2">
              10.1 FULL Mode
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>Single-case analysis. Outputs include:</p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    Critical buckling load <MathEquation math="P_{cr}" /> and
                    corresponding tip deflection{" "}
                    <MathEquation math="\delta_{cr}" />.
                  </li>
                  <li>
                    Optimal Ritz parameters{" "}
                    <MathEquation math="\alpha^*" /> and{" "}
                    <MathEquation math="\beta^*" />, and the associated
                    wavelength{" "}
                    <MathEquation math="\lambda_x^*" />.
                  </li>
                  <li>
                    Post-buckling load-deflection curve and normalized mode
                    contour (<strong>DeflectionGrid</strong>).
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* 10.2 */}
          <div className="mb-6">
            <h3 id="sens-mode" className="text-lg font-medium mb-2">
              10.2 SENS Mode
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  One-at-a-time (OAT) parameter sweeps around the baseline
                  configuration. Each selected parameter is varied within a
                  user-specified range while all others are held constant. The
                  result is a set of{" "}
                  <MathEquation math="P_{cr}" />-vs-parameter curves.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 10.3 */}
          <div className="mb-6">
            <h3 id="sobol-mode" className="text-lg font-medium mb-2">
              10.3 SOBOL Mode
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Saltelli-lite global sensitivity analysis. The total number of
                  solver evaluations is:
                </p>
                <MathEquation
                  math="N_{\text{eval}} = (2 + k)\,N_{\text{base}}"
                  display
                />
                <p>
                  where <MathEquation math="k" /> is the number of uncertain
                  parameters and <MathEquation math="N_{\text{base}}" /> is the
                  base sample size. The analysis reports first-order indices{" "}
                  <MathEquation math="S_i" /> and total-order indices{" "}
                  <MathEquation math="S_{Ti}" /> for each parameter.
                </p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 11 – Verification and Example Cases
            ================================================================ */}
        <section>
          <h2 id="verification" className="text-xl font-semibold mb-4">
            11. Verification and Example Cases
          </h2>

          {/* 11.1 */}
          <div className="mb-6">
            <h3 id="fem-benchmark" className="text-lg font-medium mb-2">
              11.1 FEM Benchmark
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  The solver has been verified against finite-element results from
                  ABAQUS. The predicted critical loads match within approximately{" "}
                  <strong>2%</strong> for all tested configurations.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* 11.2 */}
          <div className="mb-6">
            <h3 id="example-cases" className="text-lg font-medium mb-2">
              11.2 Example Cases
            </h3>
            <Card>
              <CardContent className="pt-6 prose prose-sm max-w-none">
                <p>
                  Several reference configurations are provided for validation and
                  experimentation:
                </p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    <strong>REF</strong> — Baseline reference case.
                  </li>
                  <li>
                    <strong>QI16 / QI8</strong> — Quasi-isotropic laminate
                    variants (16-ply and 8-ply).
                  </li>
                  <li>
                    <strong>CROSS</strong> — Cross-ply dominant laminate.
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* ================================================================
            Section 12 – Limitations and Recommended Use
            ================================================================ */}
        <section>
          <h2 id="limitations" className="text-xl font-semibold mb-4">
            12. Limitations and Recommended Use
          </h2>
          <Card>
            <CardContent className="pt-6 prose prose-sm max-w-none">
              <p>
                <strong>Known limitations:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>Single half-wave width mode only.</li>
                <li>Simplified edge restraint (webs treated as rotational springs).</li>
                <li>Simplified Koiter post-buckling (quartic coefficient only).</li>
                <li>
                  Thin-wall approximations for{" "}
                  <MathEquation math="EI" /> and{" "}
                  <MathEquation math="\alpha_0" />.
                </li>
                <li>No imperfection modeling.</li>
              </ul>

              <p className="mt-4">
                <strong>Recommended settings:</strong>
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  <strong>FULL mode:</strong>{" "}
                  <MathEquation math="\text{PPW} = 60" />,{" "}
                  <MathEquation math="n_{x,\min} = 1801" />.
                </li>
                <li>
                  <strong>SOBOL mode:</strong>{" "}
                  <MathEquation math="\text{PPW} = 30" />,{" "}
                  <MathEquation math="n_{x,\min} = 801" />,{" "}
                  <MathEquation math="N_{\text{base}} = 200" /> (default).
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* ================================================================
            Appendix C – Key Equations Summary
            ================================================================ */}
        <section>
          <h2 id="appendix-c" className="text-xl font-semibold mb-4">
            Appendix C: Key Equations Summary
          </h2>
          <Card>
            <CardContent className="pt-6 prose prose-sm max-w-none">
              <p>
                Collected key equations for quick reference.
              </p>

              <div className="mt-4 space-y-4">
                <div>
                  <p className="font-medium">Assumed mode shape:</p>
                  <MathEquation
                    math="w(x,y) = F(x)\,\sin\!\left(\frac{\pi\,y}{b(x)}\right), \qquad k_y(x) = \frac{\pi}{b(x)}"
                    display
                  />
                </div>

                <div>
                  <p className="font-medium">Generalized eigenproblem (M3):</p>
                  <MathEquation
                    math="\mathbf{N}\,\mathbf{c} = P\,\mathbf{D}\,\mathbf{c}, \qquad P_{cr} = \min\,\text{eigenvalue}"
                    display
                  />
                </div>

                <div>
                  <p className="font-medium">Edge-rotation stiffness:</p>
                  <MathEquation
                    math="k_{\theta,\text{web-pair}}(x) = 2 \cdot \frac{4\,D_{22}^{\text{web}}}{h(x)}"
                    display
                  />
                </div>

                <div>
                  <p className="font-medium">Load mapping weight:</p>
                  <MathEquation
                    math="\alpha_0(x) \;\propto\; \frac{L - x}{I_{\text{total}}(x)}"
                    display
                  />
                </div>

                <div>
                  <p className="font-medium">Koiter amplitude:</p>
                  <MathEquation
                    math="a(P) = \sqrt{\frac{(P - P_{cr})\,\text{Den2}}{k_4}}"
                    display
                  />
                </div>

                <div>
                  <p className="font-medium">Den2:</p>
                  <MathEquation
                    math="\text{Den2} = \int_0^L \alpha_0(x)\,\bigl[F'(x)\bigr]^2\,\frac{b(x)}{2}\,dx"
                    display
                  />
                </div>

                <div>
                  <p className="font-medium">Deflection decomposition:</p>
                  <MathEquation
                    math="\delta_{\text{lin}} = P \cdot f_b, \qquad \delta_{\text{loc}} = L \cdot \theta_{\text{fac}} \cdot a(P), \qquad \delta_{\text{tot}} = \delta_{\text{lin}} + \delta_{\text{loc}}"
                    display
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </ManualLayout>
  );
}
