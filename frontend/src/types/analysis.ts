export interface BucklingParams {
  L: number;
  b_root: number;
  b_tip: number;
  h_root: number;
  h_tip: number;
  w_f: number;
  t_face_total: number;
  face_angles: string;
  t_web_total: number;
  web_angles: string;
  Ef: number;
  Em: number;
  Gf: number;
  nuf: number;
  num: number;
  Vf: number;
  Ktheta_root_per_m: number;
  PPW: number;
  nx_min: number;
}

export interface CurveData {
  P: number[];
  delta_linear: number[];
  delta_nonlinear: number[];
  delta_total: number[];
}

export interface ContourData {
  x: number[];
  y: number[][];
  w_normalized: number[][];
  nx: number;
  ny: number;
}

export interface BucklingResults {
  core: string;
  Pcr: number;
  dcr: number;
  alpha_star: number;
  beta_star: number;
  lambda_x: number;
  curves: CurveData;
  contour: ContourData;
}

export type CoreType = "m2" | "m3";
