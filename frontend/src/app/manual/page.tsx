import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Manual | E3B Buckling",
  description: "User and Theory manuals for the E3B Buckling Analysis tool.",
};

const manuals = [
  {
    title: "User Manual",
    description:
      "Step-by-step guide for using the E3B web application: parameter input, running analysis, interpreting results, and exporting data.",
    href: "/manual/user",
  },
  {
    title: "Theory Manual",
    description:
      "Mathematical formulation behind the M2 and M3 buckling analysis cores, including Rayleigh-Ritz method, Koiter theory, and composite laminate mechanics.",
    href: "/manual/theory",
  },
];

export default function ManualIndexPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h1 className="text-2xl font-bold text-gray-900 mb-2">
        Documentation
      </h1>
      <p className="text-gray-500 mb-8">
        Select a manual to learn more about E3B.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {manuals.map((manual) => (
          <Link key={manual.href} href={manual.href}>
            <Card className="h-full hover:shadow-md transition-shadow cursor-pointer">
              <CardHeader>
                <CardTitle>{manual.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600">{manual.description}</p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
