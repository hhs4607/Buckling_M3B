"use client";

import dynamic from "next/dynamic";
import { useUncertaintyStore } from "@/stores/uncertaintyStore";
import { Card, CardContent } from "@/components/ui/card";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export function SobolChart() {
  const results = useUncertaintyStore((s) => s.results);

  if (!results) {
    return (
      <Card className="h-full">
        <CardContent className="flex items-center justify-center h-96 text-muted-foreground text-sm">
          Run Sobol analysis to see results
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Plot
          data={[
            {
              x: results.names,
              y: results.S1,
              type: "bar",
              name: "S1 (First-order)",
              marker: { color: "#3b82f6" },
            },
            {
              x: results.names,
              y: results.ST,
              type: "bar",
              name: "ST (Total)",
              marker: { color: "#f97316" },
            },
          ]}
          layout={{
            title: { text: "Sobol Sensitivity Indices", font: { size: 16 } },
            barmode: "group",
            xaxis: {
              title: { text: "Parameters" },
              tickangle: -45,
            },
            yaxis: {
              title: { text: "Sensitivity Index" },
              range: [0, 1],
            },
            height: 400,
            margin: { l: 60, r: 20, t: 60, b: 100 },
            autosize: true,
          }}
          config={{ responsive: true, displayModeBar: true }}
          className="w-full"
        />
      </CardContent>
    </Card>
  );
}
