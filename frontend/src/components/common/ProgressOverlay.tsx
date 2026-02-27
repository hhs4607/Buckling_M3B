"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";

interface ProgressOverlayProps {
  isRunning: boolean;
  progress: { current: number; total: number; message: string } | null;
  onCancel: () => void;
}

export function ProgressOverlay({
  isRunning,
  progress,
  onCancel,
}: ProgressOverlayProps) {
  const percentage = progress && progress.total > 0
    ? Math.round((progress.current / progress.total) * 100)
    : 0;

  return (
    <Dialog open={isRunning}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Analysis Running</DialogTitle>
          <DialogDescription>
            Please wait while the analysis is being processed.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3 py-2">
          <Progress value={percentage} className="h-3" />
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              {progress?.message ?? "Initializing..."}
            </span>
            <span className="font-mono font-medium">{percentage}%</span>
          </div>
          {progress && (
            <p className="text-xs text-muted-foreground text-center">
              {progress.current} / {progress.total}
            </p>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
