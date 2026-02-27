export interface SweepParam {
  key: string;
  label: string;
  enabled: boolean;
  percent: number;
  points: number;
}

export interface SensParamResult {
  param: string;
  values: number[];
  pcr_values: number[];
}

export interface SensitivityResults {
  results: SensParamResult[];
  baseline_pcr: number;
}

export interface SensProgressData {
  current: number;
  total: number;
  param: string;
  message: string;
}
