"use client";

import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useSensitivityStore } from "@/stores/sensitivityStore";
import { useAnalysisStore } from "@/stores/analysisStore";
import { AlertCircle } from "lucide-react";

export function SensControls() {
  const { core, isRunning, error, sweepParams, setCore, runSensitivity } =
    useSensitivityStore();
  const params = useAnalysisStore((s) => s.params);

  const hasEnabled = sweepParams.some((p) => p.enabled);

  return (
    <div className="flex flex-col gap-3">
      {/* Core selector */}
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium mr-1">Core:</span>
        <Button
          size="sm"
          variant={core === "m2" ? "default" : "outline"}
          onClick={() => setCore("m2")}
        >
          M2
        </Button>
        <Button
          size="sm"
          variant={core === "m3" ? "default" : "outline"}
          onClick={() => setCore("m3")}
        >
          M3
        </Button>
      </div>

      {/* Run button */}
      <Button
        onClick={() => runSensitivity(params)}
        disabled={isRunning || !hasEnabled}
        className="w-full"
        size="lg"
      >
        {isRunning ? "Running..." : "Run Sensitivity"}
      </Button>

      {/* Error message */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
