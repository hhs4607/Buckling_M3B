"use client";

import { AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ParameterInput } from "./ParameterInput";
import { PARAMETERS } from "@/constants/parameters";
import type { BucklingParams } from "@/types/analysis";

interface GeometrySectionProps {
  params: BucklingParams;
  onChange: (key: string, value: number | string) => void;
}

export function GeometrySection({ params, onChange }: GeometrySectionProps) {
  const fields = PARAMETERS.filter((p) => p.section === "geometry");

  return (
    <AccordionItem value="geometry">
      <AccordionTrigger className="text-sm font-semibold">Geometry</AccordionTrigger>
      <AccordionContent className="space-y-1 px-1">
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
