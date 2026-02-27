"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { validateField } from "@/lib/validation";
import { cn } from "@/lib/utils";

interface ParameterInputProps {
  paramKey: string;
  label: string;
  unit: string;
  tooltip: string;
  value: number | string;
  onChange: (key: string, value: number | string) => void;
}

function formatDisplayValue(value: number | string): string {
  if (typeof value === "string") return value;
  if (value === 0) return "0";
  if (Math.abs(value) >= 1e6 || Math.abs(value) < 1e-3) {
    return value.toExponential(4);
  }
  return String(value);
}

export function ParameterInput({ paramKey, label, unit, tooltip, value, onChange }: ParameterInputProps) {
  const [error, setError] = useState<string | null>(null);
  const [touched, setTouched] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [editValue, setEditValue] = useState("");
  const prevValueRef = useRef(value);

  // Sync editValue when prop value changes externally (e.g., config load)
  useEffect(() => {
    if (!isFocused && value !== prevValueRef.current) {
      setEditValue(formatDisplayValue(value));
      prevValueRef.current = value;
    }
  }, [value, isFocused]);

  const handleFocus = useCallback(() => {
    setIsFocused(true);
    // Show raw numeric value (not exponential) for easier editing
    setEditValue(String(value));
  }, [value]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const raw = e.target.value;
      setTouched(true);
      setEditValue(raw);

      // Validate for visual feedback but don't push to store
      const result = validateField(paramKey, raw);
      setError(result.valid ? null : result.message || "Invalid");

      // String params (angles) push immediately since they don't have decimal issues
      if (paramKey === "face_angles" || paramKey === "web_angles") {
        onChange(paramKey, raw);
      }
    },
    [paramKey, onChange]
  );

  const handleBlur = useCallback(() => {
    setIsFocused(false);

    // For string params, already pushed on change
    if (paramKey === "face_angles" || paramKey === "web_angles") return;

    // Convert and push to store
    const num = Number(editValue);
    if (!isNaN(num) && editValue.trim() !== "") {
      onChange(paramKey, num);
      prevValueRef.current = num;
    }
  }, [paramKey, editValue, onChange]);

  const displayValue = isFocused ? editValue : formatDisplayValue(value);

  return (
    <div className="flex items-center gap-2 py-1">
      <Tooltip>
        <TooltipTrigger asChild>
          <label className="text-sm text-gray-700 w-36 shrink-0 cursor-help truncate">
            {label}
          </label>
        </TooltipTrigger>
        <TooltipContent side="left" className="max-w-xs">
          <p className="text-xs">{tooltip}</p>
        </TooltipContent>
      </Tooltip>

      <Input
        type="text"
        value={displayValue}
        onChange={handleChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
        className={cn(
          "h-8 text-sm font-mono",
          touched && error && "border-red-500 focus-visible:ring-red-500",
          touched && !error && "border-green-500 focus-visible:ring-green-500"
        )}
      />

      {unit && (
        <span className="text-xs text-gray-500 w-12 shrink-0">{unit}</span>
      )}
    </div>
  );
}
