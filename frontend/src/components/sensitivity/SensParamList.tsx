"use client";

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { useSensitivityStore } from "@/stores/sensitivityStore";

export function SensParamList() {
  const { sweepParams, toggleParam, setPercent, setPoints } =
    useSensitivityStore();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Sweep Parameters</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Header row */}
        <div className="flex items-center gap-3 mb-2 px-1 text-xs text-muted-foreground font-medium">
          <div className="w-4" />
          <div className="flex-1">Parameter</div>
          <div className="w-16 text-right">&plusmn; %</div>
          <div className="w-16 text-right">Points</div>
        </div>

        {/* Scrollable parameter list */}
        <div className="max-h-96 overflow-y-auto space-y-1">
          {sweepParams.map((p) => (
            <div
              key={p.key}
              className="flex items-center gap-3 rounded-md px-1 py-1.5 hover:bg-muted/50 transition-colors"
            >
              <Checkbox
                checked={p.enabled}
                onCheckedChange={() => toggleParam(p.key)}
              />
              <span className="flex-1 text-sm truncate">{p.label}</span>
              <Input
                type="number"
                min={1}
                max={50}
                value={p.percent}
                onChange={(e) =>
                  setPercent(p.key, Number(e.target.value))
                }
                disabled={!p.enabled}
                className="w-16 text-right tabular-nums"
              />
              <Input
                type="number"
                min={3}
                max={20}
                value={p.points}
                onChange={(e) =>
                  setPoints(p.key, Number(e.target.value))
                }
                disabled={!p.enabled}
                className="w-16 text-right tabular-nums"
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
