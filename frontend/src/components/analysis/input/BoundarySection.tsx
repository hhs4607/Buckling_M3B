"use client";

import { AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ParameterInput } from "./ParameterInput";
import { PARAMETERS } from "@/constants/parameters";
import type { BucklingParams } from "@/types/analysis";

interface BoundarySectionProps {
  params: BucklingParams;
  onChange: (key: string, value: number | string) => void;
}

export function BoundarySection({ params, onChange }: BoundarySectionProps) {
  const fields = PARAMETERS.filter((p) => p.section === "boundary");

  return (
    <AccordionItem value="boundary">
      <AccordionTrigger className="text-sm font-semibold">Boundary Conditions</AccordionTrigger>
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
