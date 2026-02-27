"use client";

import { AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { ParameterInput } from "./ParameterInput";
import { PARAMETERS } from "@/constants/parameters";
import type { BucklingParams } from "@/types/analysis";

interface LaminateSectionProps {
  params: BucklingParams;
  onChange: (key: string, value: number | string) => void;
}

export function LaminateSection({ params, onChange }: LaminateSectionProps) {
  const faceFields = PARAMETERS.filter((p) => p.section === "face_laminate");
  const webFields = PARAMETERS.filter((p) => p.section === "web_laminate");

  return (
    <>
      <AccordionItem value="face_laminate">
        <AccordionTrigger className="text-sm font-semibold">Face Laminate</AccordionTrigger>
        <AccordionContent className="space-y-1 px-1">
          {faceFields.map((f) => (
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
      <AccordionItem value="web_laminate">
        <AccordionTrigger className="text-sm font-semibold">Web Laminate</AccordionTrigger>
        <AccordionContent className="space-y-1 px-1">
          {webFields.map((f) => (
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
    </>
  );
}
