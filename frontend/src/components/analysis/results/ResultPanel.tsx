"use client";

import { useAnalysisStore } from "@/stores/analysisStore";
import { DescriptionCard } from "./DescriptionCard";
import { ResultSummary } from "./ResultSummary";
import { LoadDeflectionChart } from "./LoadDeflectionChart";
import { ModeContourChart } from "./ModeContourChart";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

export function ResultPanel() {
  const { results, isRunning, error } = useAnalysisStore();

  return (
    <div className="space-y-4">
      <DescriptionCard />

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {isRunning && (
        <div className="space-y-3">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      )}

      {results && !isRunning && (
        <>
          <ResultSummary results={results} />
          <LoadDeflectionChart results={results} />
          <ModeContourChart results={results} />
        </>
      )}

      {!results && !isRunning && !error && (
        <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
          Run an analysis to see results
        </div>
      )}
    </div>
  );
}
