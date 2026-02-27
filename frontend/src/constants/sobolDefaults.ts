export interface SobolDefault {
  key: string;
  label: string;
  low: number;
  high: number;
}

export const SOBOL_DEFAULTS: SobolDefault[] = [
  { key: "L", label: "Beam Length (m)", low: 1.35, high: 1.65 },
  { key: "b_root", label: "Root Width (m)", low: 0.072, high: 0.088 },
  { key: "b_tip", label: "Tip Width (m)", low: 0.036, high: 0.044 },
  { key: "h_root", label: "Root Core Height (m)", low: 0.0225, high: 0.0275 },
  { key: "h_tip", label: "Tip Core Height (m)", low: 0.0135, high: 0.0165 },
  { key: "w_f", label: "Flange Width (m)", low: 0.018, high: 0.022 },
  { key: "t_face_total", label: "Face Thickness (m)", low: 0.0018, high: 0.0022 },
  { key: "t_web_total", label: "Web Thickness (m)", low: 0.00135, high: 0.00165 },
  { key: "Ef", label: "Fiber Modulus (Pa)", low: 207e9, high: 253e9 },
  { key: "Em", label: "Matrix Modulus (Pa)", low: 3.15e9, high: 3.85e9 },
  { key: "Gf", label: "Fiber Shear Modulus (Pa)", low: 81e9, high: 99e9 },
  { key: "nuf", label: "Fiber Poisson Ratio", low: 0.18, high: 0.22 },
  { key: "num", label: "Matrix Poisson Ratio", low: 0.315, high: 0.385 },
  { key: "Vf", label: "Fiber Volume Fraction", low: 0.54, high: 0.66 },
  { key: "Ktheta_root_per_m", label: "Root Spring (N·m/m)", low: 0.9e9, high: 1.1e9 },
  { key: "PPW", label: "Points/Wavelength", low: 54, high: 66 },
  { key: "nx_min", label: "Min Grid Points", low: 1621, high: 1981 },
];
