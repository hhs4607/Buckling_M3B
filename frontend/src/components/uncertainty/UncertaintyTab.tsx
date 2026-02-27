"use client";

import { useUncertaintyStore } from "@/stores/uncertaintyStore";
import { SobolParamList } from "./SobolParamList";
import { SobolControls } from "./SobolControls";
import { SobolChart } from "./SobolChart";
import { ProgressOverlay } from "@/components/common/ProgressOverlay";

export function UncertaintyTab() {
  const { isRunning, progress, cancel } = useUncertaintyStore();

  return (
    <div className="flex flex-col gap-4 md:flex-row">
      {/* Left panel: parameter list + controls */}
      <div className="space-y-4 md:w-96 shrink-0">
        <SobolParamList />
        <SobolControls />
      </div>

      {/* Right panel: chart */}
      <div className="flex-1 min-w-0">
        <SobolChart />
      </div>

      {/* Progress overlay dialog */}
      <ProgressOverlay
        isRunning={isRunning}
        progress={
          progress
            ? {
                current: progress.current,
                total: progress.total,
                message: progress.message,
              }
            : null
        }
        onCancel={cancel}
      />
    </div>
  );
}
