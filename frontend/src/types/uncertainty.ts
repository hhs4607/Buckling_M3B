export interface UncertainParam {
  key: string;
  label: string;
  enabled: boolean;
  low: number;
  high: number;
}

export interface SobolResults {
  names: string[];
  S1: number[];
  ST: number[];
}

export interface SobolProgressData {
  current: number;
  total: number;
  phase: string;
  message: string;
}
