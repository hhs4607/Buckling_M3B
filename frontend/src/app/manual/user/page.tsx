import { Card, CardContent } from "@/components/ui/card";
import { ManualLayout } from "@/components/manual/ManualLayout";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "User Manual | E3B Buckling",
  description:
    "Comprehensive user manual for the E3B Buckling Analysis web application — covering single-case analysis, sensitivity studies, and Sobol-based uncertainty quantification.",
};

const sections = [
  { id: "overview", title: "1. Overview" },
  { id: "what-the-tool-solves", title: "2. What the Tool Solves" },
  { id: "models-m2-vs-m3", title: "3. Models: M2 vs M3" },
  { id: "gui-overview", title: "4. GUI Overview" },
  { id: "quick-start", title: "5. Quick Start (3 Minutes)" },
  { id: "quick-start-single", title: "5.1 Single Case", level: 2 },
  { id: "quick-start-sensitivity", title: "5.2 Sensitivity (OAT)", level: 2 },
  { id: "quick-start-uncertainty", title: "5.3 Uncertainty (Sobol)", level: 2 },
  { id: "inputs-baseline", title: "6. Inputs (Baseline)" },
  { id: "inputs-geometry", title: "6.1 Geometry", level: 2 },
  { id: "inputs-laminate", title: "6.2 Laminate Definitions", level: 2 },
  { id: "inputs-materials", title: "6.3 Materials", level: 2 },
  { id: "solver-settings", title: "7. Solver / Numerical Settings" },
  { id: "solver-root-spring", title: "7.1 Root Spring", level: 2 },
  { id: "solver-ppw", title: "7.2 PPW", level: 2 },
  { id: "solver-nx-min", title: "7.3 nx_min", level: 2 },
  { id: "solver-n-base", title: "7.4 N_base", level: 2 },
  { id: "outputs", title: "8. Outputs" },
  { id: "sensitivity-oat", title: "9. Sensitivity (OAT)" },
  { id: "uncertainty-sobol", title: "10. Uncertainty (Sobol UQ)" },
  { id: "verification", title: "11. Verification & Examples" },
  { id: "limitations", title: "12. Limitations" },
  { id: "appendix-a", title: "Appendix A: Input Key Reference" },
];

export default function UserManualPage() {
  return (
    <ManualLayout title="User Manual" sections={sections}>
      <div className="space-y-10">
        {/* ---------------------------------------------------------------- */}
        {/* 1. Overview                                                       */}
        {/* ---------------------------------------------------------------- */}
        <section id="overview">
          <h2>1. Overview</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                <strong>E3B_M3</strong> is a GUI-based engineering tool for
                estimating the buckling critical load (P<sub>cr</sub>) and
                related mode characteristics for an <em>equivalent box-beam</em>{" "}
                (E3B) representation of a wing / beam-like structure. It
                supports:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  <strong>Single Case Buckling analysis</strong> — baseline run
                </li>
                <li>
                  <strong>One-at-a-Time (OAT) Sensitivity studies</strong>
                </li>
                <li>
                  <strong>Global Uncertainty Quantification</strong> — Sobol
                  indices
                </li>
              </ul>
              <p className="mt-3">
                The web app is organized into three main tabs:{" "}
                <strong>Buckling</strong> (Single Case),{" "}
                <strong>Sensitivity</strong> (OAT), and{" "}
                <strong>Uncertainty</strong> (Sobol UQ).
              </p>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 2. What the Tool Solves                                           */}
        {/* ---------------------------------------------------------------- */}
        <section id="what-the-tool-solves">
          <h2>2. What the Tool Solves</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                In early-stage design and trade studies, you often need to
                answer:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  &ldquo;Which configuration is more
                  buckling-resistant?&rdquo;
                </li>
                <li>
                  &ldquo;Which parameter actually drives P<sub>cr</sub>
                  ?&rdquo;
                </li>
                <li>
                  &ldquo;If inputs vary &plusmn;X%, how uncertain is P
                  <sub>cr</sub>?&rdquo;
                </li>
                <li>
                  &ldquo;Should I use a simplified model (M2) or a richer model
                  (M3)?&rdquo;
                </li>
              </ul>
              <p className="mt-3">
                E3B_M3 is built for <strong>rapid iteration</strong>, not for
                replacing full-fidelity FE buckling analyses. It provides:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  Fast estimates of P<sub>cr</sub>
                </li>
                <li>A consistent way to compare configurations</li>
                <li>Sensitivity ranking and Sobol-based influence measures</li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 3. Models: M2 vs M3                                               */}
        {/* ---------------------------------------------------------------- */}
        <section id="models-m2-vs-m3">
          <h2>3. Models: M2 vs M3</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <h3>M2 (Simplified)</h3>
              <p>
                <strong>Use when:</strong> fast sweeps, robust behavior,
                ranking / trend validation.
              </p>
              <p>
                <strong>Characteristics:</strong> lower computational cost, fewer
                coupling effects, more stable for broad sweeps.
              </p>

              <h3 className="mt-4">M3 (Enhanced)</h3>
              <p>
                <strong>Use when:</strong> improved physics fidelity
                (bending-shear coupling), near boundary of validity, final trade
                studies before FE.
              </p>
              <p>
                <strong>Characteristics:</strong> higher computational cost, more
                parameters matter, sensitive to grid density.
              </p>

              <div className="mt-4 rounded-md border border-blue-200 bg-blue-50 p-4 text-blue-900">
                <p className="font-semibold">Recommended Default</p>
                <p className="mt-1">
                  Start with <strong>M2</strong> during exploration. Switch to{" "}
                  <strong>M3</strong> for final comparative assessment.
                </p>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 4. GUI Overview                                                   */}
        {/* ---------------------------------------------------------------- */}
        <section id="gui-overview">
          <h2>4. GUI Overview (Web App Layout)</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                Each tab follows the same structural pattern:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  <strong>Left panel</strong> — inputs and settings
                </li>
                <li>
                  <strong>Right panel</strong> — plots, progress messages, and
                  results summary
                </li>
                <li>
                  <strong>Run buttons</strong> — execute the selected analysis
                  mode
                </li>
              </ul>

              <h3 className="mt-4">Three Tabs</h3>
              <table className="mt-2 w-full text-sm">
                <thead>
                  <tr>
                    <th className="text-left pr-4 pb-2">Tab</th>
                    <th className="text-left pb-2">Purpose</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="pr-4 py-1 font-medium">
                      Buckling (Single Case)
                    </td>
                    <td className="py-1">
                      Computes P<sub>cr</sub> + mode + plots
                    </td>
                  </tr>
                  <tr>
                    <td className="pr-4 py-1 font-medium">Sensitivity (OAT)</td>
                    <td className="py-1">
                      Varies one parameter at a time
                    </td>
                  </tr>
                  <tr>
                    <td className="pr-4 py-1 font-medium">
                      Uncertainty (Sobol UQ)
                    </td>
                    <td className="py-1">Computes Sobol indices</td>
                  </tr>
                </tbody>
              </table>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 5. Quick Start (3 Minutes)                                        */}
        {/* ---------------------------------------------------------------- */}
        <section id="quick-start">
          <h2>5. Quick Start (3 Minutes)</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                The following three walkthroughs demonstrate the core workflows.
                Each can be completed in about one minute.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* 5.1 Single Case */}
        <section id="quick-start-single">
          <h3>5.1 Single Case (Baseline)</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <ol className="list-decimal list-inside space-y-2">
                <li>
                  Open the <strong>Buckling</strong> tab.
                </li>
                <li>Enter geometry parameters (L, b, h, w_f).</li>
                <li>Enter laminate definitions (thickness + ply angles).</li>
                <li>Enter material properties (Ef, Em, Gf, nuf, num, Vf).</li>
                <li>
                  Choose solver model &mdash; start with <strong>M2</strong>.
                </li>
                <li>
                  Click <strong>Run Buckling</strong>.
                </li>
                <li>
                  Review P<sub>cr</sub> and plots in the right panel.
                </li>
                <li>Export results if needed.</li>
              </ol>
            </CardContent>
          </Card>
        </section>

        {/* 5.2 Sensitivity (OAT) */}
        <section id="quick-start-sensitivity">
          <h3>5.2 Sensitivity (OAT)</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <ol className="list-decimal list-inside space-y-2">
                <li>
                  Go to the <strong>Sensitivity</strong> tab.
                </li>
                <li>Choose the parameters to sweep.</li>
                <li>
                  Set range (<strong>&plusmn;10%</strong>) and number of points (
                  <strong>5</strong>).
                </li>
                <li>Select the solver model.</li>
                <li>
                  Click <strong>Run Sensitivity</strong>.
                </li>
                <li>
                  Read which parameter changes P<sub>cr</sub> the most.
                </li>
              </ol>
            </CardContent>
          </Card>
        </section>

        {/* 5.3 Uncertainty (Sobol UQ) */}
        <section id="quick-start-uncertainty">
          <h3>5.3 Uncertainty (Sobol UQ)</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <ol className="list-decimal list-inside space-y-2">
                <li>
                  Go to the <strong>Uncertainty</strong> tab.
                </li>
                <li>
                  Set Low / High bounds for each parameter, or click{" "}
                  <strong>Update from Baseline</strong> to auto-populate bounds
                  with a uniform uncertainty percentage.
                </li>
                <li>
                  Set <strong>N_base</strong> (base sample count) and{" "}
                  <strong>Seed</strong>.
                </li>
                <li>Select the model (M2 or M3).</li>
                <li>
                  Click <strong>Run Sobol</strong>.
                </li>
                <li>
                  Read <strong>S1</strong> (direct effect) and{" "}
                  <strong>ST</strong> (direct + interactions) for each
                  parameter.
                </li>
              </ol>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 6. Inputs (Baseline)                                              */}
        {/* ---------------------------------------------------------------- */}
        <section id="inputs-baseline">
          <h2>6. Inputs (Baseline)</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                All analysis modes share the same baseline input set. Inputs are
                grouped into three categories: Geometry, Laminate Definitions,
                and Materials.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* 6.1 Geometry */}
        <section id="inputs-geometry">
          <h3>6.1 Geometry</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <ul className="list-disc list-inside space-y-2">
                <li>
                  <strong>L</strong> &mdash; Beam length from root to tip (m).
                  Defines the overall span of the equivalent box-beam. Longer
                  beams tend to lower P<sub>cr</sub>.
                </li>
                <li>
                  <strong>b_root / b_tip</strong> &mdash; Width at root and tip
                  (m). Controls the taper ratio in the width direction.
                  Increasing root width raises P<sub>cr</sub>; a larger
                  root-to-tip ratio increases the taper effect.
                </li>
                <li>
                  <strong>h_root / h_tip</strong> &mdash; Core height at root and
                  tip (m). Controls the taper ratio in the height direction.
                  Greater height increases bending stiffness and raises P
                  <sub>cr</sub>.
                </li>
                <li>
                  <strong>w_f</strong> &mdash; Flange width (m). Defines the
                  width of the flange elements in the box-beam cross section.
                  Affects local buckling behavior.
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* 6.2 Laminate Definitions */}
        <section id="inputs-laminate">
          <h3>6.2 Laminate Definitions</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <h4 className="font-medium">Face Laminate</h4>
              <ul className="list-disc list-inside space-y-1 mt-1">
                <li>
                  <strong>t_face_total</strong> &mdash; Total face laminate
                  thickness (m)
                </li>
                <li>
                  <strong>face_angles</strong> &mdash; Ply angles for the face
                  laminate (degrees, comma-separated)
                </li>
              </ul>

              <h4 className="font-medium mt-4">Web Laminate</h4>
              <ul className="list-disc list-inside space-y-1 mt-1">
                <li>
                  <strong>t_web_total</strong> &mdash; Total web laminate
                  thickness (m)
                </li>
                <li>
                  <strong>web_angles</strong> &mdash; Ply angles for the web
                  laminate (degrees, comma-separated)
                </li>
              </ul>

              <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-4 text-amber-900">
                <p className="font-semibold">Input Rules</p>
                <p className="mt-1">
                  Angles are specified in degrees, comma-separated. Spaces are
                  allowed between values (e.g.,{" "}
                  <code>0, 45, -45, 90</code>).
                </p>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* 6.3 Materials */}
        <section id="inputs-materials">
          <h3>6.3 Materials</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <ul className="list-disc list-inside space-y-2">
                <li>
                  <strong>Ef</strong> &mdash; Fiber modulus (Pa). Longitudinal
                  stiffness of the fiber material.
                </li>
                <li>
                  <strong>Em</strong> &mdash; Matrix modulus (Pa). Stiffness of
                  the surrounding matrix material.
                </li>
                <li>
                  <strong>Gf</strong> &mdash; Fiber shear modulus (Pa). Shear
                  stiffness of the fiber.
                </li>
                <li>
                  <strong>nuf</strong> &mdash; Fiber Poisson&apos;s ratio.
                  Typically around 0.2&ndash;0.3.
                </li>
                <li>
                  <strong>num</strong> &mdash; Matrix Poisson&apos;s ratio.
                  Typically around 0.3&ndash;0.4.
                </li>
                <li>
                  <strong>Vf</strong> &mdash; Fiber volume fraction (0&ndash;1).
                  Fraction of the composite occupied by fiber. Higher Vf
                  increases stiffness and generally raises P<sub>cr</sub>.
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 7. Solver / Numerical Settings                                    */}
        {/* ---------------------------------------------------------------- */}
        <section id="solver-settings">
          <h2>7. Solver / Numerical Settings</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                These parameters control the numerical resolution and boundary
                conditions of the analysis. Incorrect settings can lead to missed
                critical modes or unnecessarily long run times.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* 7.1 Root Spring */}
        <section id="solver-root-spring">
          <h3>7.1 Root Spring (Ktheta_root_per_m)</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                Rotational stiffness at the root boundary (N&middot;m/m). This
                parameter models the degree of rotational restraint at the
                fixed end of the beam.
              </p>
              <p className="mt-2">
                Tuning the root spring can significantly change P<sub>cr</sub>.
                A very high value approximates a clamped condition; a lower
                value represents a more flexible support.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* 7.2 PPW */}
        <section id="solver-ppw">
          <h3>7.2 PPW (Points Per Wavelength)</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                Controls the spatial resolution of the discretization relative
                to the expected buckling wavelength.
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  <strong>Default:</strong> 24
                </li>
                <li>
                  <strong>Too low</strong> &rarr; missed critical mode or
                  inaccurate P<sub>cr</sub>
                </li>
                <li>
                  <strong>Too high</strong> &rarr; slower computation with
                  diminishing accuracy gains
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* 7.3 nx_min */}
        <section id="solver-nx-min">
          <h3>7.3 nx_min (Minimum Grid Points)</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                Minimum number of grid points along the beam length. Prevents
                under-resolved geometry, especially for beams with high taper
                ratios.
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  <strong>Default:</strong> 200
                </li>
              </ul>
              <p className="mt-2">
                Increasing nx_min beyond the default is only necessary when
                geometry changes are very rapid along the span.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* 7.4 N_base */}
        <section id="solver-n-base">
          <h3>7.4 N_base (Sobol Base Sample Count)</h3>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                The base sample count used by the Saltelli sampling scheme in
                the Sobol UQ analysis. Total function evaluations scale as{" "}
                <code>N_base &times; (2k + 2)</code>, where <em>k</em> is the
                number of uncertain parameters.
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  <strong>Default:</strong> 1024 &mdash; stable, publication-ready
                  reporting
                </li>
                <li>
                  <strong>Fast check:</strong> 256 &mdash; quick screening with
                  wider confidence intervals
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 8. Outputs and How to Read Them                                   */}
        {/* ---------------------------------------------------------------- */}
        <section id="outputs">
          <h2>8. Outputs and How to Read Them</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <h3>Core Outputs</h3>
              <ul className="list-disc list-inside space-y-1 mt-1">
                <li>
                  <strong>P<sub>cr</sub></strong> &mdash; Critical buckling load
                </li>
                <li>
                  <strong>Mode indicator / wavelength proxy</strong> &mdash;
                  Characterizes the buckling pattern
                </li>
                <li>
                  <strong>Plots</strong> &mdash; Visual response curves and mode
                  shapes
                </li>
              </ul>

              <h3 className="mt-4">What &ldquo;Good&rdquo; Looks Like</h3>
              <ul className="list-disc list-inside space-y-1 mt-1">
                <li>Clear minimum in the response curve</li>
                <li>
                  Consistent behavior when changing PPW / nx_min (convergence)
                </li>
                <li>No dramatic mode swapping between adjacent cases</li>
              </ul>

              <h3 className="mt-4">Exported Artifacts</h3>
              <ul className="list-disc list-inside space-y-1 mt-1">
                <li>Results table (numerical summary)</li>
                <li>Plots (PNG / interactive)</li>
                <li>Run log (parameter echo and solver messages)</li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 9. Sensitivity (OAT)                                              */}
        {/* ---------------------------------------------------------------- */}
        <section id="sensitivity-oat">
          <h2>9. Sensitivity (OAT)</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <h3>Purpose</h3>
              <p>
                &ldquo;If I change one parameter &plusmn;X%, how much does P
                <sub>cr</sub> move?&rdquo; OAT sensitivity is best for:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>Ranking which parameters drive the output</li>
                <li>Detecting nonlinearities in the response</li>
                <li>
                  Finding parameters that trigger mode-switching behavior
                </li>
              </ul>

              <h3 className="mt-4">Recommended Setup</h3>
              <table className="mt-2 w-full text-sm">
                <thead>
                  <tr>
                    <th className="text-left pr-4 pb-2">Setting</th>
                    <th className="text-left pb-2">Recommended Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="pr-4 py-1">Range</td>
                    <td className="py-1">&plusmn;10%</td>
                  </tr>
                  <tr>
                    <td className="pr-4 py-1">Points</td>
                    <td className="py-1">5</td>
                  </tr>
                  <tr>
                    <td className="pr-4 py-1">Model (screening)</td>
                    <td className="py-1">M2</td>
                  </tr>
                  <tr>
                    <td className="pr-4 py-1">Model (final)</td>
                    <td className="py-1">M3</td>
                  </tr>
                </tbody>
              </table>

              <h3 className="mt-4">Interpreting Results</h3>
              <ul className="list-disc list-inside space-y-2 mt-2">
                <li>
                  <strong>Monotonic slope</strong> &mdash; consistent,
                  predictable effect on P<sub>cr</sub>.
                </li>
                <li>
                  <strong>Nonlinear curve</strong> &mdash; strong model physics
                  interaction; the parameter&apos;s influence depends on its
                  value.
                </li>
                <li>
                  <strong>Discontinuous jump</strong> &mdash; mode switch. Verify
                  with higher PPW / nx_min to confirm it is not a numerical
                  artifact.
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 10. Uncertainty (Sobol UQ)                                        */}
        {/* ---------------------------------------------------------------- */}
        <section id="uncertainty-sobol">
          <h2>10. Uncertainty (Sobol UQ)</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <h3>Purpose</h3>
              <p>
                Answer two key questions: &ldquo;Which parameters dominate
                output uncertainty?&rdquo; and &ldquo;Are interactions
                important?&rdquo;
              </p>

              <h3 className="mt-4">Bounds Setup</h3>
              <p>
                Set <strong>Low</strong> and <strong>High</strong> bounds for
                each parameter manually, or use the{" "}
                <strong>&ldquo;Update from Baseline&rdquo;</strong> button to
                automatically populate bounds using a uniform uncertainty
                percentage around the baseline values.
              </p>

              <h3 className="mt-4">Reading Sobol Indices</h3>
              <ul className="list-disc list-inside space-y-2 mt-2">
                <li>
                  <strong>S1 (first-order)</strong> &mdash; Effect of the
                  parameter <em>alone</em>, ignoring all interactions.
                </li>
                <li>
                  <strong>ST (total-order)</strong> &mdash; Effect of the
                  parameter <em>including</em> all interactions with other
                  parameters.
                </li>
              </ul>

              <div className="mt-4 rounded-md border border-gray-200 bg-gray-50 p-4 text-gray-800">
                <p className="font-semibold">How to Interpret the Indices</p>
                <ul className="list-disc list-inside space-y-1 mt-2">
                  <li>
                    <strong>High S1 and High ST</strong> &rarr; Dominant
                    independent driver.
                  </li>
                  <li>
                    <strong>Low S1 but High ST</strong> &rarr; Interactions
                    matter; the parameter&apos;s effect depends on the values of
                    other parameters.
                  </li>
                  <li>
                    <strong>Low S1 and Low ST</strong> &rarr; Parameter has
                    negligible influence on output uncertainty.
                  </li>
                </ul>
              </div>

              <h3 className="mt-4">Reproducibility</h3>
              <p>
                Fix the <strong>Seed</strong> to produce repeatable Sobol
                indices across runs. Changing the seed will produce a different
                quasi-random sampling sequence, which may shift index estimates
                slightly.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 11. Verification & Example Cases                                  */}
        {/* ---------------------------------------------------------------- */}
        <section id="verification">
          <h2>11. Verification &amp; Example Cases</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <p>
                The application ships with several regression-check cases that
                can be loaded as presets:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  <strong>REF</strong> &mdash; Reference configuration
                </li>
                <li>
                  <strong>QI16</strong> &mdash; Quasi-isotropic, 16-ply
                </li>
                <li>
                  <strong>QI8</strong> &mdash; Quasi-isotropic, 8-ply
                </li>
                <li>
                  <strong>CROSS</strong> &mdash; Cross-ply laminate
                </li>
              </ul>
              <p className="mt-3">
                For each example case, compare the following against known
                reference values:
              </p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>
                  P<sub>cr</sub> error (should be within acceptable tolerance)
                </li>
                <li>Mode similarity (same buckling pattern as the reference)</li>
                <li>
                  Stability of results when varying PPW and nx_min
                  (convergence check)
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* 12. Limitations & Recommended Use                                 */}
        {/* ---------------------------------------------------------------- */}
        <section id="limitations">
          <h2>12. Limitations &amp; Recommended Use</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none">
              <h3>Known Limitations</h3>
              <ul className="list-disc list-inside space-y-2 mt-2">
                <li>
                  The equivalent-beam representation cannot capture all local 3D
                  effects (e.g., skin wrinkling, stiffener crippling).
                </li>
                <li>
                  Boundary conditions are simplified &mdash; only root spring
                  stiffness is modeled; complex multi-point constraints are not
                  supported.
                </li>
                <li>
                  Laminate behavior is reduced to an effective representation via
                  Classical Lamination Theory; ply-level damage or
                  delamination is not modeled.
                </li>
              </ul>

              <h3 className="mt-4">Recommended Defaults</h3>
              <table className="mt-2 w-full text-sm">
                <thead>
                  <tr>
                    <th className="text-left pr-4 pb-2">Parameter</th>
                    <th className="text-left pb-2">Recommended Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="pr-4 py-1">PPW</td>
                    <td className="py-1">24</td>
                  </tr>
                  <tr>
                    <td className="pr-4 py-1">nx_min</td>
                    <td className="py-1">200</td>
                  </tr>
                  <tr>
                    <td className="pr-4 py-1">N_base</td>
                    <td className="py-1">
                      1024 (production) / 256 (quick check)
                    </td>
                  </tr>
                </tbody>
              </table>
            </CardContent>
          </Card>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* Appendix A: Input Key Reference                                   */}
        {/* ---------------------------------------------------------------- */}
        <section id="appendix-a">
          <h2>Appendix A: Input Key Reference</h2>
          <Card>
            <CardContent className="prose prose-sm max-w-none overflow-x-auto">
              <table className="mt-2 w-full text-sm">
                <thead>
                  <tr>
                    <th className="text-left pr-3 pb-2">Key</th>
                    <th className="text-left pr-3 pb-2">Unit</th>
                    <th className="text-left pr-3 pb-2">Description</th>
                    <th className="text-left pb-2">Used In</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">L</td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Beam length (root to tip)</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">b_root</td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Width at root</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">b_tip</td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Width at tip</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">h_root</td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Core height at root</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">h_tip</td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Core height at tip</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">w_f</td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Flange width</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">
                      t_face_total
                    </td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Face laminate thickness</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">
                      face_angles
                    </td>
                    <td className="pr-3 py-1.5">deg</td>
                    <td className="pr-3 py-1.5">
                      Face laminate ply angles (comma-separated)
                    </td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">
                      t_web_total
                    </td>
                    <td className="pr-3 py-1.5">m</td>
                    <td className="pr-3 py-1.5">Web laminate thickness</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">
                      web_angles
                    </td>
                    <td className="pr-3 py-1.5">deg</td>
                    <td className="pr-3 py-1.5">
                      Web laminate ply angles (comma-separated)
                    </td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">Ef</td>
                    <td className="pr-3 py-1.5">Pa</td>
                    <td className="pr-3 py-1.5">Fiber elastic modulus</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">Em</td>
                    <td className="pr-3 py-1.5">Pa</td>
                    <td className="pr-3 py-1.5">Matrix elastic modulus</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">Gf</td>
                    <td className="pr-3 py-1.5">Pa</td>
                    <td className="pr-3 py-1.5">Fiber shear modulus</td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">nuf</td>
                    <td className="pr-3 py-1.5">&mdash;</td>
                    <td className="pr-3 py-1.5">
                      Fiber Poisson&apos;s ratio
                    </td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">num</td>
                    <td className="pr-3 py-1.5">&mdash;</td>
                    <td className="pr-3 py-1.5">
                      Matrix Poisson&apos;s ratio
                    </td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">Vf</td>
                    <td className="pr-3 py-1.5">&mdash;</td>
                    <td className="pr-3 py-1.5">
                      Fiber volume fraction (0&ndash;1)
                    </td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">
                      Ktheta_root_per_m
                    </td>
                    <td className="pr-3 py-1.5">N&middot;m/m</td>
                    <td className="pr-3 py-1.5">
                      Rotational spring stiffness at root
                    </td>
                    <td className="py-1.5">All</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">PPW</td>
                    <td className="pr-3 py-1.5">&mdash;</td>
                    <td className="pr-3 py-1.5">Points per wavelength</td>
                    <td className="py-1.5">Buckling, Sensitivity</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">nx_min</td>
                    <td className="pr-3 py-1.5">&mdash;</td>
                    <td className="pr-3 py-1.5">
                      Minimum grid points along span
                    </td>
                    <td className="py-1.5">Buckling, Sensitivity</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">N_base</td>
                    <td className="pr-3 py-1.5">&mdash;</td>
                    <td className="pr-3 py-1.5">Sobol base sample count</td>
                    <td className="py-1.5">Uncertainty</td>
                  </tr>
                  <tr>
                    <td className="pr-3 py-1.5 font-mono text-xs">Seed</td>
                    <td className="pr-3 py-1.5">&mdash;</td>
                    <td className="pr-3 py-1.5">
                      Random seed for reproducibility
                    </td>
                    <td className="py-1.5">Uncertainty</td>
                  </tr>
                </tbody>
              </table>
            </CardContent>
          </Card>
        </section>
      </div>
    </ManualLayout>
  );
}
