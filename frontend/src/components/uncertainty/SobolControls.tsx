"use client";

import { useState } from "react";
import { useUncertaintyStore } from "@/stores/uncertaintyStore";
import { useAnalysisStore } from "@/stores/analysisStore";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import type { CoreType } from "@/types/analysis";

export function SobolControls() {
  const {
    nBase,
    seed,
    core,
    isRunning,
    error,
    uncertainParams,
    setNBase,
    setSeed,
    setCore,
    runSobol,
    updateFromBaseline,
  } = useUncertaintyStore();

  const analysisParams = useAnalysisStore((s) => s.params);
  const [uncertaintyPct, setUncertaintyPct] = useState(10);

  const hasEnabledParams = uncertainParams.some((p) => p.enabled);

  const handleRun = () => {
    runSobol(analysisParams);
  };

  const handleUpdateFromBaseline = () => {
    updateFromBaseline(analysisParams, uncertaintyPct);
  };

  return (
    <Card>
      <CardContent className="space-y-3">
        {/* Row 1: N_base + Seed */}
        <div className="flex items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">
              N (base)
            </label>
            <Input
              type="number"
              className="w-24"
              min={10}
              max={10000}
              value={nBase}
              onChange={(e) => setNBase(Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">
              Seed
            </label>
            <Input
              type="number"
              className="w-24"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
        </div>

        {/* Row 2: Uncertainty % + Update from Baseline */}
        <div className="flex items-end gap-3">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">
              &plusmn;%
            </label>
            <Input
              type="number"
              className="w-20"
              min={1}
              max={100}
              value={uncertaintyPct}
              onChange={(e) => setUncertaintyPct(Number(e.target.value))}
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleUpdateFromBaseline}
          >
            Update from Baseline
          </Button>
        </div>

        {/* Row 3: Core selector + Run button */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 rounded-md border p-0.5">
            {(["m2", "m3"] as CoreType[]).map((c) => (
              <button
                key={c}
                type="button"
                className={cn(
                  "rounded px-3 py-1 text-xs font-medium transition-colors",
                  core === c
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => setCore(c)}
              >
                {c === "m2" ? "M2" : "M3"}
              </button>
            ))}
          </div>
          <Button
            className="flex-1"
            disabled={isRunning || !hasEnabledParams}
            onClick={handleRun}
          >
            {isRunning ? "Running..." : "Run Sobol"}
          </Button>
        </div>

        {/* Error display */}
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
