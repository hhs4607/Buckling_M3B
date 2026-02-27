"use client";

import { AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ParameterInput } from "./ParameterInput";
import { PARAMETERS } from "@/constants/parameters";
import type { BucklingParams, CoreType } from "@/types/analysis";

interface SolverSectionProps {
  params: BucklingParams;
  core: CoreType;
  onChange: (key: string, value: number | string) => void;
  onCoreChange: (core: CoreType) => void;
}

export function SolverSection({ params, core, onChange, onCoreChange }: SolverSectionProps) {
  const fields = PARAMETERS.filter((p) => p.section === "solver");

  return (
    <AccordionItem value="solver">
      <AccordionTrigger className="text-sm font-semibold">Solver</AccordionTrigger>
      <AccordionContent className="space-y-3 px-1">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-gray-700">Core:</span>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="radio"
              name="core"
              value="m3"
              checked={core === "m3"}
              onChange={() => onCoreChange("m3")}
              className="accent-blue-600"
            />
            <span className="text-sm">M3 (Accurate)</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="radio"
              name="core"
              value="m2"
              checked={core === "m2"}
              onChange={() => onCoreChange("m2")}
              className="accent-blue-600"
            />
            <span className="text-sm">M2 (Fast)</span>
          </label>
        </div>
        {fields.map((f) => (
          <ParameterInput
            key={f.key}
            paramKey={f.key}
            label={f.label}
            unit={f.unit}
            tooltip={f.tooltip}
            value={(params as unknown as Record<string, number | string>)[f.key]}
            onChange={onChange}
          />
        ))}
      </AccordionContent>
    </AccordionItem>
  );
}
