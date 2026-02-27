"use client";

import dynamic from "next/dynamic";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { useSensitivityStore } from "@/stores/sensitivityStore";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export function SensitivityChart() {
  const results = useSensitivityStore((s) => s.results);

  if (!results) {
    return (
      <Card className="flex-1">
        <CardContent className="flex items-center justify-center min-h-[400px]">
          <p className="text-muted-foreground text-sm">
            Run sensitivity analysis to see results
          </p>
        </CardContent>
      </Card>
    );
  }

  const { results: paramResults, baseline_pcr } = results;

  return (
    <Card className="flex-1">
      <CardHeader>
        <CardTitle className="text-base">Sensitivity Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {paramResults.map((r) => (
            <div key={r.param}>
              <Plot
                data={[
                  {
                    x: r.values,
                    y: r.pcr_values,
                    type: "scatter" as const,
                    mode: "lines+markers" as const,
                    name: r.param,
                    line: { color: "#3b82f6" },
                    marker: { size: 5 },
                  },
                  {
                    x: [r.values[0], r.values[r.values.length - 1]],
                    y: [baseline_pcr, baseline_pcr],
                    type: "scatter" as const,
                    mode: "lines" as const,
                    name: "Baseline",
                    line: { color: "#ef4444", dash: "dash" as const, width: 2 },
                  },
                ]}
                layout={{
                  title: { text: r.param, font: { size: 13 } },
                  xaxis: { title: { text: r.param } },
                  yaxis: { title: { text: "Pcr (N)" } },
                  margin: { l: 60, r: 20, t: 40, b: 40 },
                  height: 250,
                  showlegend: false,
                  autosize: true,
                }}
                config={{ responsive: true, displayModeBar: false }}
                useResizeHandler
                className="w-full"
                style={{ height: "250px" }}
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
