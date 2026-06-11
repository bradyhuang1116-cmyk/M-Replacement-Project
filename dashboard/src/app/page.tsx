"use client";

import { useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import ReplacingCard from "@/components/ReplacingCard";
import FileTable from "@/components/FileTable";
import GPUWidget from "@/components/GPUWidget";
import { useJobStatus } from "@/hooks/useJobStatus";
import { useGPUInfo } from "@/hooks/useGPUInfo";
import { apiFetch } from "@/lib/api";

export default function DashboardPage() {
  const { job, files } = useJobStatus();
  const { data: gpu } = useGPUInfo();

  const handleStart = useCallback(() => {
    window.location.href = "/datasets";
  }, []);

  const handleStop = useCallback(async () => {
    try {
      await apiFetch("/system/stop-all", { method: "POST" });
    } catch (e) {
      console.error("Failed to stop job:", e);
    }
  }, []);

  return (
    <div className="flex h-screen bg-[rgb(10,10,10)]">
      <Sidebar />

      <div className="flex-1 flex overflow-hidden">
        <main className="flex-1 overflow-y-auto p-6 space-y-5">
          <h1 className="text-lg font-semibold text-[rgb(245,245,245)]">
            Dashboard
          </h1>
          <ReplacingCard job={job} onStart={handleStart} onStop={handleStop} />
          <FileTable files={files} />
        </main>

        <aside className="w-80 border-l border-[rgb(38,38,38)] overflow-y-auto p-4 space-y-4 shrink-0">
          <h2 className="text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider px-1">
            Hardware Monitor
          </h2>
          {gpu ? (
            <GPUWidget gpu={gpu} />
          ) : (
            <div className="bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] p-4 text-center text-sm text-[rgb(115,115,115)]">
              Connecting to GPU...
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
