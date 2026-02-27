"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatNumber } from "@/lib/utils";
import type { BucklingResults } from "@/types/analysis";

interface ResultSummaryProps {
  results: BucklingResults;
}

export function ResultSummary({ results }: ResultSummaryProps) {
  const items = [
    { label: "Pcr", value: formatNumber(results.Pcr, 2), unit: "N" },
    { label: "Pcr", value: formatNumber(results.Pcr / 1000, 4), unit: "kN" },
    { label: "dcr", value: formatNumber(results.dcr * 1000, 2), unit: "mm" },
    { label: "α*", value: formatNumber(results.alpha_star, 2), unit: "1/m" },
    { label: "β*", value: formatNumber(results.beta_star, 2), unit: "1/m" },
    { label: "λx*", value: formatNumber(results.lambda_x * 1000, 2), unit: "mm" },
  ];

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Results</CardTitle>
          <Badge variant="outline">{results.core}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {items.map((item, i) => (
            <div key={i} className="text-center p-2 bg-gray-50 rounded">
              <p className="text-xs text-gray-500">{item.label}</p>
              <p className="text-sm font-mono font-semibold">{item.value}</p>
              <p className="text-xs text-gray-400">{item.unit}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
