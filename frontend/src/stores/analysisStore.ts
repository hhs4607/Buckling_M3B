import { create } from "zustand";
import type { BucklingParams, BucklingResults, CoreType } from "@/types/analysis";
import { DEFAULT_PARAMS } from "@/constants/defaults";
import { apiPost } from "@/lib/api";

interface AnalysisStore {
  params: BucklingParams;
  core: CoreType;
  results: BucklingResults | null;
  isRunning: boolean;
  error: string | null;

  setParam: (key: string, value: number | string) => void;
  setAllParams: (params: BucklingParams) => void;
  setCore: (core: CoreType) => void;
  runAnalysis: () => Promise<void>;
  loadConfig: (config: Partial<BucklingParams> & { core?: CoreType }) => void;
  exportConfig: () => { core: CoreType; params: BucklingParams };
}

export const useAnalysisStore = create<AnalysisStore>((set, get) => ({
  params: { ...DEFAULT_PARAMS },
  core: "m3",
  results: null,
  isRunning: false,
  error: null,

  setParam: (key, value) =>
    set((state) => ({
      params: { ...state.params, [key]: value },
    })),

  setAllParams: (params) => set({ params }),

  setCore: (core) => set({ core }),

  runAnalysis: async () => {
    set({ isRunning: true, error: null });
    try {
      const { core, params } = get();
      const results = await apiPost<BucklingResults>("/api/buckling/run", {
        core,
        params,
      });
      set({ results, isRunning: false });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Analysis failed",
        isRunning: false,
      });
    }
  },

  loadConfig: (config) => {
    const { core, ...params } = config;
    if (core) set({ core });
    set((state) => ({
      params: { ...state.params, ...params },
    }));
  },

  exportConfig: () => {
    const { core, params } = get();
    return { core, params };
  },
}));
