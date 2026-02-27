"use client";

import { useState, useCallback } from "react";
import { useUncertaintyStore } from "@/stores/uncertaintyStore";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface FloatInputProps {
  value: number;
  disabled: boolean;
  onCommit: (value: number) => void;
}

function FloatInput({ value, disabled, onCommit }: FloatInputProps) {
  const [editValue, setEditValue] = useState(String(value));
  const [isFocused, setIsFocused] = useState(false);

  const handleFocus = useCallback(() => {
    setIsFocused(true);
    setEditValue(String(value));
  }, [value]);

  const handleBlur = useCallback(() => {
    setIsFocused(false);
    const num = Number(editValue);
    if (!isNaN(num) && editValue.trim() !== "") {
      onCommit(num);
    }
  }, [editValue, onCommit]);

  const displayValue = isFocused ? editValue : String(value);

  return (
    <Input
      type="text"
      className="w-24 text-right text-sm font-mono"
      value={displayValue}
      disabled={disabled}
      onChange={(e) => setEditValue(e.target.value)}
      onFocus={handleFocus}
      onBlur={handleBlur}
    />
  );
}

export function SobolParamList() {
  const { uncertainParams, toggleParam, setLow, setHigh } =
    useUncertaintyStore();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Uncertain Parameters</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Column headers */}
        <div className="flex items-center gap-3 mb-2 px-1">
          <div className="w-4" />
          <span className="flex-1 text-xs font-medium text-muted-foreground">
            Parameter
          </span>
          <span className="w-24 text-right text-xs font-medium text-muted-foreground">
            Low
          </span>
          <span className="w-24 text-right text-xs font-medium text-muted-foreground">
            High
          </span>
        </div>

        {/* Scrollable parameter list */}
        <div className="max-h-96 overflow-y-auto space-y-1">
          {uncertainParams.map((p) => (
            <div
              key={p.key}
              className="flex items-center gap-3 rounded-md px-1 py-1.5 hover:bg-muted/50"
            >
              <Checkbox
                checked={p.enabled}
                onCheckedChange={() => toggleParam(p.key)}
              />
              <span
                className={cn(
                  "flex-1 text-sm truncate",
                  !p.enabled && "text-muted-foreground"
                )}
              >
                {p.label}
              </span>
              <FloatInput
                value={p.low}
                disabled={!p.enabled}
                onCommit={(v) => setLow(p.key, v)}
              />
              <FloatInput
                value={p.high}
                disabled={!p.enabled}
                onCommit={(v) => setHigh(p.key, v)}
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
