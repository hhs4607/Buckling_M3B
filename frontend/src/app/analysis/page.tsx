import { AnalysisTabs } from "@/components/analysis/AnalysisTabs";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Analysis | E3B Buckling",
  description:
    "Buckling analysis of double-tapered composite box beams with sensitivity and uncertainty quantification.",
};

export default function AnalysisPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">
          Buckling Analysis
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Configure beam parameters and run M2/M3 buckling analysis
        </p>
      </div>
      <AnalysisTabs />
    </div>
  );
}
