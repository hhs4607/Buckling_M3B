"use client";

import { SensParamList } from "./SensParamList";
import { SensControls } from "./SensControls";
import { SensitivityChart } from "./SensitivityChart";
import { ProgressOverlay } from "@/components/common/ProgressOverlay";
import { useSensitivityStore } from "@/stores/sensitivityStore";

export function SensitivityTab() {
  const { isRunning, progress, cancel } = useSensitivityStore();

  return (
    <div className="flex flex-col md:flex-row gap-4 p-4">
      {/* Left panel: parameter list + controls */}
      <div className="md:w-80 flex flex-col gap-4 shrink-0">
        <SensParamList />
        <SensControls />
      </div>

      {/* Right panel: chart results */}
      <div className="flex-1 min-w-0">
        <SensitivityChart />
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
