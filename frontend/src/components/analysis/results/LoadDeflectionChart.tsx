"use client";

import dynamic from "next/dynamic";
import type { BucklingResults } from "@/types/analysis";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface LoadDeflectionChartProps {
  results: BucklingResults;
}

export function LoadDeflectionChart({ results }: LoadDeflectionChartProps) {
  const { curves, Pcr, dcr } = results;

  // Convert to mm for display
  const toMm = (arr: number[]) => arr.map((v) => v * 1000);

  return (
    <Plot
      data={[
        {
          x: toMm(curves.delta_linear),
          y: curves.P,
          mode: "lines",
          name: "Linear",
          line: { color: "steelblue", dash: "dash", width: 2 },
        },
        {
          x: toMm(curves.delta_nonlinear),
          y: curves.P,
          mode: "lines",
          name: "Nonlinear",
          line: { color: "orange", dash: "dot", width: 2 },
        },
        {
          x: toMm(curves.delta_total),
          y: curves.P,
          mode: "lines",
          name: "Total",
          line: { color: "green", width: 2.5 },
        },
        {
          x: [dcr * 1000],
          y: [Pcr],
          mode: "markers",
          name: `Pcr = ${Pcr.toFixed(1)} N`,
          marker: { color: "red", size: 10, symbol: "diamond" },
        },
      ]}
      layout={{
        title: { text: "Load\u2013Deflection Curves", font: { size: 14 } },
        xaxis: { title: { text: "Deflection \u03B4 [mm]" } },
        yaxis: { title: { text: "Load P [N]" } },
        legend: { x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.8)" },
        margin: { l: 60, r: 20, t: 40, b: 50 },
        autosize: true,
      }}
      config={{ responsive: true, displayModeBar: true, toImageButtonOptions: { format: "png", filename: "load_deflection" } }}
      useResizeHandler
      className="w-full"
      style={{ height: "350px" }}
    />
  );
}
