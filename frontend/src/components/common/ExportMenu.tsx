"use client";

import { Button } from "@/components/ui/button";
import { useAnalysisStore } from "@/stores/analysisStore";
import { formatNumber } from "@/lib/utils";
import { cn } from "@/lib/utils";

interface ExportMenuProps {
  className?: string;
}

export function ExportMenu({ className }: ExportMenuProps) {
  const results = useAnalysisStore((s) => s.results);

  const handleExport = () => {
    if (!results) return;

    const content = [
      "E3B Buckling Analysis Results",
      "==============================",
      `Core: ${results.core}`,
      `Pcr: ${formatNumber(results.Pcr, 4)} N`,
      `dcr: ${formatNumber(results.dcr, 4)} m`,
      `alpha*: ${formatNumber(results.alpha_star, 4)}`,
      `beta*: ${formatNumber(results.beta_star, 4)}`,
      `lambda_x: ${formatNumber(results.lambda_x, 4)}`,
    ].join("\n");

    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "e3b-results.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={cn(className)}>
      <Button
        variant="outline"
        size="sm"
        onClick={handleExport}
        disabled={!results}
      >
        Export TXT
      </Button>
    </div>
  );
}
