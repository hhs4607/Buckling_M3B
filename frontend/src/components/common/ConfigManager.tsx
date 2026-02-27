"use client";

import { useRef } from "react";
import { Button } from "@/components/ui/button";
import { useAnalysisStore } from "@/stores/analysisStore";
import { cn } from "@/lib/utils";

interface ConfigManagerProps {
  className?: string;
}

export function ConfigManager({ className }: ConfigManagerProps) {
  const exportConfig = useAnalysisStore((s) => s.exportConfig);
  const loadConfig = useAnalysisStore((s) => s.loadConfig);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSave = () => {
    const config = exportConfig();
    const blob = new Blob([JSON.stringify(config, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "e3b-config.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleLoad = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const config = JSON.parse(event.target?.result as string);
        loadConfig(config);
      } catch {
        console.error("Failed to parse config file");
      }
    };
    reader.readAsText(file);

    // Reset input so the same file can be loaded again
    e.target.value = "";
  };

  return (
    <div className={cn("flex gap-2", className)}>
      <Button variant="outline" size="sm" onClick={handleSave}>
        Save
      </Button>
      <Button variant="outline" size="sm" onClick={handleLoad}>
        Load
      </Button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
