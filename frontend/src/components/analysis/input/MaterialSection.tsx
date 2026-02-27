"use client";

import { AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ParameterInput } from "./ParameterInput";
import { PARAMETERS } from "@/constants/parameters";
import type { BucklingParams } from "@/types/analysis";

interface MaterialSectionProps {
  params: BucklingParams;
  onChange: (key: string, value: number | string) => void;
}

export function MaterialSection({ params, onChange }: MaterialSectionProps) {
  const fields = PARAMETERS.filter((p) => p.section === "material");

  return (
    <AccordionItem value="material">
      <AccordionTrigger className="text-sm font-semibold">Materials</AccordionTrigger>
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
