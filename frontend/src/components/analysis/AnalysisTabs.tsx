"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { InputPanel } from "./input/InputPanel";
import { ResultPanel } from "./results/ResultPanel";
import { SensitivityTab } from "@/components/sensitivity/SensitivityTab";
import { UncertaintyTab } from "@/components/uncertainty/UncertaintyTab";
import { ConfigManager } from "@/components/common/ConfigManager";
import { ExportMenu } from "@/components/common/ExportMenu";

export function AnalysisTabs() {
  return (
    <Tabs defaultValue="buckling" className="w-full">
      <div className="flex items-center justify-between gap-4 mb-4">
        <TabsList>
          <TabsTrigger value="buckling">Buckling</TabsTrigger>
          <TabsTrigger value="sensitivity">Sensitivity</TabsTrigger>
          <TabsTrigger value="uncertainty">Uncertainty</TabsTrigger>
        </TabsList>

        <div className="hidden sm:flex items-center gap-2">
          <ConfigManager />
          <ExportMenu />
        </div>
      </div>

      <TabsContent value="buckling">
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="w-full lg:w-80 xl:w-96 shrink-0">
            <InputPanel />
          </div>
          <div className="flex-1 min-w-0">
            <ResultPanel />
          </div>
        </div>
      </TabsContent>

      <TabsContent value="sensitivity">
        <SensitivityTab />
      </TabsContent>

      <TabsContent value="uncertainty">
        <UncertaintyTab />
      </TabsContent>

      {/* Mobile config/export buttons */}
      <div className="sm:hidden flex items-center gap-2 mt-4 pt-4 border-t">
        <ConfigManager />
        <ExportMenu />
      </div>
    </Tabs>
  );
}
