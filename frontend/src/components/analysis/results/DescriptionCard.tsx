"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function DescriptionCard() {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Double Tapered Composite Box Beam: Buckling Analysis</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-gray-600 space-y-1">
        <p>Ritz energy method with Koiter post-buckling theory for local skin buckling.</p>
        <p className="text-xs text-gray-400">
          M3: 2-term Ritz with root rotational spring | M2: 1-term fast approximation
        </p>
      </CardContent>
    </Card>
  );
}
