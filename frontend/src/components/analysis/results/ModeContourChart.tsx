"use client";

import dynamic from "next/dynamic";
import type { BucklingResults } from "@/types/analysis";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface ModeContourChartProps {
  results: BucklingResults;
}

export function ModeContourChart({ results }: ModeContourChartProps) {
  const { contour } = results;

  return (
    <Plot
      data={[
        {
          z: contour.w_normalized,
          x: contour.x,
          y: contour.y[0],
          type: "heatmap",
          colorscale: "Viridis",
          colorbar: { title: { text: "w/max|w|", side: "right" } },
          hovertemplate: "x: %{x:.3f} m<br>y: %{y:.4f} m<br>w/max|w|: %{z:.3f}<extra></extra>",
        },
      ]}
      layout={{
        title: { text: "Mode Contour", font: { size: 14 } },
        xaxis: { title: { text: "x [m]" } },
        yaxis: { title: { text: "y [m]" } },
        margin: { l: 60, r: 20, t: 40, b: 50 },
        autosize: true,
      }}
      config={{ responsive: true, displayModeBar: true, toImageButtonOptions: { format: "png", filename: "mode_contour" } }}
      useResizeHandler
      className="w-full"
      style={{ height: "300px" }}
    />
  );
}
