export interface ValidationResult {
  valid: boolean;
  message?: string;
}

const POSITIVE_FIELDS = [
  "L", "b_root", "b_tip", "h_root", "h_tip", "w_f",
  "t_face_total", "t_web_total", "Ef", "Em", "Gf",
];

export function validateField(key: string, value: string): ValidationResult {
  if (!value.trim()) return { valid: false, message: "Required" };

  if (key === "face_angles" || key === "web_angles") {
    const parts = value.replace(/\s/g, "").split(",");
    if (parts.length === 0 || parts.some((p) => isNaN(Number(p)) || p === "")) {
      return { valid: false, message: "Comma-separated numbers required" };
    }
    return { valid: true };
  }

  const num = Number(value);
  if (isNaN(num)) return { valid: false, message: "Must be a number" };

  if (POSITIVE_FIELDS.includes(key)) {
    if (num <= 0) return { valid: false, message: "Must be > 0" };
  } else if (key === "Vf") {
    if (num <= 0 || num >= 1) return { valid: false, message: "Must be between 0 and 1" };
  } else if (key === "nuf" || key === "num") {
    if (num <= -1 || num >= 0.5) return { valid: false, message: "Must be between -1 and 0.5" };
  } else if (key === "Ktheta_root_per_m") {
    if (num < 0) return { valid: false, message: "Must be >= 0" };
  } else if (key === "PPW" || key === "nx_min") {
    if (num < 10) return { valid: false, message: "Must be >= 10" };
  }

  return { valid: true };
}
