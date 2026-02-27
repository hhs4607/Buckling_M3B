import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(value: number, decimals = 4): string {
  if (Math.abs(value) >= 1e6 || (Math.abs(value) < 1e-3 && value !== 0)) {
    return value.toExponential(decimals);
  }
  return value.toFixed(decimals);
}
