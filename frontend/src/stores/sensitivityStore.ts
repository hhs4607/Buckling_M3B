import { create } from "zustand";
import type { CoreType, BucklingParams } from "@/types/analysis";
import type { SweepParam, SensitivityResults, SensProgressData } from "@/types/sensitivity";
import { SWEEPABLE_PARAMS } from "@/constants/parameters";
import { apiPost, sseUrl } from "@/lib/api";
import { connectSSE } from "@/lib/sse";

interface SensitivityStore {
  sweepParams: SweepParam[];
  core: CoreType;
  isRunning: boolean;
  progress: SensProgressData | null;
  results: SensitivityResults | null;
  error: string | null;
  cancelFn: (() => void) | null;

  toggleParam: (key: string) => void;
  setPercent: (key: string, pct: number) => void;
  setPoints: (key: string, pts: number) => void;
  setCore: (core: CoreType) => void;
  runSensitivity: (baselineParams: BucklingParams) => Promise<void>;
  cancel: () => void;
}

export const useSensitivityStore = create<SensitivityStore>((set, get) => ({
  sweepParams: SWEEPABLE_PARAMS.map((p) => ({
    key: p.key,
    label: p.label,
    enabled: false,
    percent: 10,
    points: 5,
  })),
  core: "m2",
  isRunning: false,
  progress: null,
  results: null,
  error: null,
  cancelFn: null,

  toggleParam: (key) =>
    set((state) => ({
      sweepParams: state.sweepParams.map((p) =>
        p.key === key ? { ...p, enabled: !p.enabled } : p
      ),
    })),

  setPercent: (key, pct) =>
    set((state) => ({
      sweepParams: state.sweepParams.map((p) =>
        p.key === key ? { ...p, percent: pct } : p
      ),
    })),

  setPoints: (key, pts) =>
    set((state) => ({
      sweepParams: state.sweepParams.map((p) =>
        p.key === key ? { ...p, points: pts } : p
      ),
    })),

  setCore: (core) => set({ core }),

  runSensitivity: async (baselineParams) => {
    const { sweepParams, core } = get();
    const enabled = sweepParams.filter((p) => p.enabled);
    if (enabled.length === 0) {
      set({ error: "Select at least one parameter" });
      return;
    }

    set({ isRunning: true, error: null, progress: null, results: null });

    try {
      const res = await apiPost<{ job_id: string; stream_url: string }>(
        "/api/sensitivity/run",
        {
          core,
          baseline_params: baselineParams,
          sweep_params: enabled.map((p) => ({
            key: p.key,
            percent: p.percent,
            points: p.points,
          })),
        }
      );

      const cancel = connectSSE<SensProgressData, SensitivityResults>(
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

  cancel: () => {
    const { cancelFn } = get();
    cancelFn?.();
    set({ isRunning: false, cancelFn: null });
  },
}));
