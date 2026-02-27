"use client";

import { Accordion } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { GeometrySection } from "./GeometrySection";
import { LaminateSection } from "./LaminateSection";
import { MaterialSection } from "./MaterialSection";
import { BoundarySection } from "./BoundarySection";
import { SolverSection } from "./SolverSection";
import { useAnalysisStore } from "@/stores/analysisStore";

export function InputPanel() {
  const { params, core, isRunning, setParam, setCore, runAnalysis } = useAnalysisStore();

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto pr-2">
        <Accordion type="multiple" defaultValue={["geometry", "face_laminate", "web_laminate", "material", "boundary", "solver"]}>
          <GeometrySection params={params} onChange={setParam} />
          <LaminateSection params={params} onChange={setParam} />
          <MaterialSection params={params} onChange={setParam} />
          <BoundarySection params={params} onChange={setParam} />
          <SolverSection params={params} core={core} onChange={setParam} onCoreChange={setCore} />
        </Accordion>
      </div>

      <div className="pt-4 border-t mt-4">
        <Button
          onClick={runAnalysis}
          disabled={isRunning}
          className="w-full"
          size="lg"
        >
          {isRunning ? "Running..." : "Run Analysis"}
        </Button>
      </div>
    </div>
  );
}
