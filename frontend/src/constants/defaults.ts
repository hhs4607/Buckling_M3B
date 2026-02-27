import type { BucklingParams } from "@/types/analysis";

export const DEFAULT_PARAMS: BucklingParams = {
  L: 1.5,
  b_root: 0.08,
  b_tip: 0.04,
  h_root: 0.025,
  h_tip: 0.015,
  w_f: 0.02,
  t_face_total: 0.002,
  face_angles: "0,45,-45,90",
  t_web_total: 0.0015,
  web_angles: "0,90",
  Ef: 230e9,
  Em: 3.5e9,
  Gf: 90e9,
  nuf: 0.2,
  num: 0.35,
  Vf: 0.6,
  Ktheta_root_per_m: 1e9,
  PPW: 60,
  nx_min: 1801,
};
