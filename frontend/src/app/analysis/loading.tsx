import { Skeleton } from "@/components/ui/skeleton";

export default function AnalysisLoading() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div className="mb-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-4 w-80 mt-2" />
      </div>
      <Skeleton className="h-10 w-96 mb-4" />
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="w-full lg:w-80 xl:w-96 shrink-0 space-y-3">
          {[...Array(5)].map((_, i) => (
            <Skeleton key={i} className="h-16" />
          ))}
        </div>
        <div className="flex-1 space-y-4">
          <Skeleton className="h-32" />
          <Skeleton className="h-64" />
        </div>
      </div>
    </div>
  );
}
