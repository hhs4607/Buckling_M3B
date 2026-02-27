import { create } from "zustand";
import type { CoreType, BucklingParams } from "@/types/analysis";
import type { UncertainParam, SobolResults, SobolProgressData } from "@/types/uncertainty";
import { SOBOL_DEFAULTS } from "@/constants/sobolDefaults";
import { apiPost, sseUrl } from "@/lib/api";
import { connectSSE } from "@/lib/sse";

interface UncertaintyStore {
  uncertainParams: UncertainParam[];
  nBase: number;
  seed: number;
  core: CoreType;
  isRunning: boolean;
  progress: SobolProgressData | null;
  results: SobolResults | null;
  error: string | null;
  cancelFn: (() => void) | null;

  toggleParam: (key: string) => void;
  setLow: (key: string, value: number) => void;
  setHigh: (key: string, value: number) => void;
  setNBase: (n: number) => void;
  setSeed: (s: number) => void;
  setCore: (core: CoreType) => void;
  runSobol: (baselineParams: BucklingParams) => Promise<void>;
  updateFromBaseline: (params: BucklingParams, pct: number) => void;
  cancel: () => void;
}

export const useUncertaintyStore = create<UncertaintyStore>((set, get) => ({
  uncertainParams: SOBOL_DEFAULTS.map((d) => ({
    key: d.key,
    label: d.label,
    enabled: false,
    low: d.low,
    high: d.high,
  })),
  nBase: 100,
  seed: 1234,
  core: "m2",
  isRunning: false,
  progress: null,
  results: null,
  error: null,
  cancelFn: null,

  toggleParam: (key) =>
    set((state) => ({
      uncertainParams: state.uncertainParams.map((p) =>
        p.key === key ? { ...p, enabled: !p.enabled } : p
      ),
    })),

  setLow: (key, value) =>
    set((state) => ({
      uncertainParams: state.uncertainParams.map((p) =>
        p.key === key ? { ...p, low: value } : p
      ),
    })),

  setHigh: (key, value) =>
    set((state) => ({
      uncertainParams: state.uncertainParams.map((p) =>
        p.key === key ? { ...p, high: value } : p
      ),
    })),

  setNBase: (n) => set({ nBase: n }),
  setSeed: (s) => set({ seed: s }),
  setCore: (core) => set({ core }),

  runSobol: async (baselineParams) => {
    const { uncertainParams, nBase, seed, core } = get();
    const enabled = uncertainParams.filter((p) => p.enabled);
    if (enabled.length === 0) {
      set({ error: "Select at least one parameter" });
      return;
    }

    set({ isRunning: true, error: null, progress: null, results: null });

    try {
      const res = await apiPost<{ job_id: string; stream_url: string }>(
        "/api/sobol/run",
        {
          core,
          baseline_params: baselineParams,
          uncertain_params: enabled.map((p) => ({
            key: p.key,
            low: p.low,
            high: p.high,
          })),
          n_base: nBase,
          seed,
        }
      );

      const cancel = connectSSE<SobolProgressData, SobolResults>(
        sseUrl(res.stream_url),
        {
          onProgress: (data) => set({ progress: data }),
          onResult: (data) => set({ results: data }),
          onDone: () => set({ isRunning: false, cancelFn: null }),
          onError: (err) => set({ error: err.message, isRunning: false, cancelFn: null }),
        }
      );
      set({ cancelFn: cancel });
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : "Failed to start",
        isRunning: false,
      });
    }
  },

  updateFromBaseline: (params, pct) => {
    const fraction = pct / 100;
    set((state) => ({
      uncertainParams: state.uncertainParams.map((p) => {
        const baseline = (params as unknown as Record<string, number | string>)[p.key];
        if (typeof baseline === "number") {
          return {
            ...p,
            low: baseline * (1 - fraction),
            high: baseline * (1 + fraction),
          };
        }
        return p;
      }),
    }));
  },

  cancel: () => {
    const { cancelFn } = get();
    cancelFn?.();
    set({ isRunning: false, cancelFn: null });
  },
}));
