export interface ParamMeta {
  key: string;
  label: string;
  unit: string;
  tooltip: string;
  section: "geometry" | "face_laminate" | "web_laminate" | "material" | "boundary" | "solver";
}

export const PARAMETERS: ParamMeta[] = [
  { key: "L", label: "Beam Length", unit: "m", tooltip: "Beam length from root to tip (meters)", section: "geometry" },
  { key: "b_root", label: "Root Width", unit: "m", tooltip: "Beam width at root (clamped end) in meters", section: "geometry" },
  { key: "b_tip", label: "Tip Width", unit: "m", tooltip: "Beam width at tip (free end) in meters", section: "geometry" },
  { key: "h_root", label: "Root Core Height", unit: "m", tooltip: "Core/foam height at root in meters", section: "geometry" },
  { key: "h_tip", label: "Tip Core Height", unit: "m", tooltip: "Core/foam height at tip in meters", section: "geometry" },
  { key: "w_f", label: "Flange Width", unit: "m", tooltip: "Width of each vertical web flange in meters", section: "geometry" },
  { key: "t_face_total", label: "Total Thickness", unit: "m", tooltip: "Total thickness of face laminate in meters", section: "face_laminate" },
  { key: "face_angles", label: "Ply Angles", unit: "deg", tooltip: "Ply angles for face laminate (comma-separated)", section: "face_laminate" },
  { key: "t_web_total", label: "Total Thickness", unit: "m", tooltip: "Total thickness of web laminate in meters", section: "web_laminate" },
  { key: "web_angles", label: "Ply Angles", unit: "deg", tooltip: "Ply angles for web laminate (comma-separated)", section: "web_laminate" },
  { key: "Ef", label: "Fiber Modulus", unit: "Pa", tooltip: "Fiber longitudinal elastic modulus in Pascals", section: "material" },
  { key: "Em", label: "Matrix Modulus", unit: "Pa", tooltip: "Matrix elastic modulus in Pascals", section: "material" },
  { key: "Gf", label: "Fiber Shear Modulus", unit: "Pa", tooltip: "Fiber shear modulus in Pascals", section: "material" },
  { key: "nuf", label: "Fiber Poisson Ratio", unit: "", tooltip: "Fiber Poisson's ratio (typically 0.2-0.3)", section: "material" },
  { key: "num", label: "Matrix Poisson Ratio", unit: "", tooltip: "Matrix Poisson's ratio (typically 0.3-0.4)", section: "material" },
  { key: "Vf", label: "Fiber Volume Fraction", unit: "", tooltip: "Fiber volume fraction (0 < Vf < 1, typically 0.5-0.7)", section: "material" },
  { key: "Ktheta_root_per_m", label: "Root Spring", unit: "N·m/m", tooltip: "Rotational spring stiffness at root per unit width", section: "boundary" },
  { key: "PPW", label: "Points/Wavelength", unit: "", tooltip: "Points per wavelength for spatial discretization", section: "solver" },
  { key: "nx_min", label: "Min Grid Points", unit: "", tooltip: "Minimum number of grid points along beam length", section: "solver" },
];

export const SWEEPABLE_PARAMS = PARAMETERS.map((p) => ({
  key: p.key,
  label: `${p.label}${p.unit ? ` (${p.unit})` : ""}`,
}));
