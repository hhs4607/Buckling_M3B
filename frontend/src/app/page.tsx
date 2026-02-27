import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "E3B Buckling Analysis | Home",
  description:
    "Web-based buckling analysis tool for double-tapered composite box beams with sensitivity and uncertainty quantification.",
};

const features = [
  {
    title: "Buckling Analysis",
    description:
      "Compute critical buckling loads using M2 (fast) or M3 (accurate) formulations for double-tapered composite box beams.",
    icon: (
      <svg className="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
      </svg>
    ),
  },
  {
    title: "Sensitivity Analysis",
    description:
      "One-at-a-time parameter sweeps to understand how each design variable affects the critical buckling load.",
    icon: (
      <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z" />
      </svg>
    ),
  },
  {
    title: "Uncertainty Quantification",
    description:
      "Sobol sensitivity indices (S1 and ST) via Saltelli-lite sampling to quantify parameter uncertainty contributions.",
    icon: (
      <svg className="h-8 w-8 text-orange-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
      </svg>
    ),
  },
];

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-blue-50 to-white py-16 sm:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 tracking-tight">
            E3B Buckling Analysis
          </h1>
          <p className="mt-4 text-lg sm:text-xl text-gray-600 max-w-2xl mx-auto">
            Web-based tool for buckling analysis of double-tapered composite box beams
            with integrated sensitivity and uncertainty quantification.
          </p>
          <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/analysis">
              <Button size="lg" className="text-base px-8">
                Start Analysis
              </Button>
            </Link>
            <Link href="/manual">
              <Button size="lg" variant="outline" className="text-base px-8">
                View Manual
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {features.map((feature) => (
              <Card key={feature.title} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="mb-2">{feature.icon}</div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Description Section */}
      <section className="bg-gray-50 py-16">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            About E3B
          </h2>
          <p className="text-gray-600 leading-relaxed">
            The E3B (Embraer Efficient Engineering of Beams) tool computes critical buckling
            loads and mode shapes for double-tapered, closed-section composite box beams under
            compressive loading. It uses a semi-analytical approach based on the Rayleigh-Ritz
            method with Koiter&apos;s initial post-buckling theory for imperfection sensitivity.
          </p>
        </div>
      </section>
    </div>
  );
}
