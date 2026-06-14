"use client";

import { useState, useEffect, useRef } from "react";
import type { JobState } from "@/types";
import { Info, Play, Square, ChevronDown, ChevronUp, Cloud, HardDrive } from "lucide-react";
import { getMode, setMode, type InferenceMode } from "@/lib/api";

interface Props {
  job: JobState;
  onStart: () => void;
  onStop: () => void;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

function phaseLabel(phase: string): string | null {
  switch (phase) {
    case "starting_vlm":
      return "Starting VLM Service...";
    case "warming_up":
      return "Warming Up Models...";
    default:
      return null;
  }
}

export default function ReplacingCard({ job, onStart, onStop }: Props) {
  const [elapsed, setElapsed] = useState(job.elapsedSeconds);
  const [logsOpen, setLogsOpen] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const [mode, setModeState] = useState<InferenceMode>("cloud");
  const [modeBusy, setModeBusy] = useState(false);

  useEffect(() => {
    getMode().then(setModeState).catch(() => {});
  }, []);

  const toggleMode = async () => {
    if (modeBusy) return;
    const next: InferenceMode = mode === "cloud" ? "local" : "cloud";
    setModeBusy(true);
    try {
      const applied = await setMode(next);
      setModeState(applied);
    } catch {
      // 保持原状
    } finally {
      setModeBusy(false);
    }
  };

  useEffect(() => {
    setElapsed(job.elapsedSeconds);
  }, [job.elapsedSeconds]);

  useEffect(() => {
    if (!job.isRunning || job.phase !== "processing") return;
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [job.isRunning, job.phase]);

  useEffect(() => {
    if (logsOpen && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [job.logs, logsOpen]);

  const isActive = job.isRunning || job.phase === "stopping";
  const pLabel = phaseLabel(job.phase);
  const progress = job.totalFiles > 0
    ? Math.round(
        (job.phase === "processing"
          ? job.currentFile / job.totalFiles
          : job.phase === "warming_up"
          ? 0.05
          : 0.02) * 100
      )
    : 0;

  return (
    <div className="bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4">
        <div className="flex items-center gap-2">
          <Info size={18} className="text-[rgb(163,163,163)]" />
          <h2 className="text-[rgb(245,245,245)] font-semibold text-base">
            {isActive ? "Replacing" : "Idle"}
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {/* 本地 / 云端 模式开关（运行中禁用，处理前可切） */}
          <button
            onClick={toggleMode}
            disabled={isActive || modeBusy}
            title={isActive ? "运行中无法切换模式" : `当前：${mode === "cloud" ? "云端" : "本地"}模式，点击切换`}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
              isActive || modeBusy
                ? "opacity-50 cursor-not-allowed bg-white/5 text-slate-400 border-white/10"
                : mode === "cloud"
                ? "cursor-pointer bg-sky-500/15 text-sky-400 border-sky-500/30 hover:bg-sky-500/25"
                : "cursor-pointer bg-amber-500/15 text-amber-400 border-amber-500/30 hover:bg-amber-500/25"
            }`}
          >
            {mode === "cloud" ? <Cloud size={11} /> : <HardDrive size={11} />}
            {mode === "cloud" ? "云端" : "本地"}
          </button>

          {/* Run / Running badge */}
          {!isActive ? (
            <button
              onClick={onStart}
              className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium cursor-pointer bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/25 transition-colors"
            >
              <Play size={10} fill="currentColor" />
              run
            </button>
          ) : (
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              running
            </span>
          )}

          {/* Stop button */}
          {isActive && (
            <button
              onClick={onStop}
              className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium cursor-pointer bg-rose-500/15 text-rose-400 border border-rose-500/30 hover:bg-rose-500/25 transition-colors"
            >
              <Square size={10} fill="currentColor" />
              stop
            </button>
          )}
        </div>
      </div>

      {/* Progress section */}
      {isActive && (
        <div className="px-5 pb-5">
          {/* Progress label */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-[rgb(212,212,212)]">
              {pLabel ? `Progress: ${pLabel}` : "Progress"}
            </span>
            {job.phase === "processing" && (
              <span className="text-sm text-[rgb(163,163,163)]">
                File {job.currentFile} of {job.totalFiles}
              </span>
            )}
          </div>

          {/* Progress bar */}
          <div className="w-full h-2 bg-[rgb(38,38,38)] rounded-full overflow-hidden mb-4">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                pLabel ? "bg-amber-500 animate-pulse" : "bg-blue-500"
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Elapsed time */}
          {job.phase === "processing" && (
            <div className="flex items-center gap-2 mb-3">
              <div className="flex items-center gap-1.5 text-[rgb(163,163,163)]">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                <span className="text-xs">Elapsed Time</span>
              </div>
              <span className="text-sm text-[rgb(229,229,229)] font-mono">
                {formatElapsed(elapsed)}
              </span>
            </div>
          )}

          {/* Logs toggle */}
          <button
            onClick={() => setLogsOpen((o) => !o)}
            className="flex items-center gap-1.5 text-xs text-[rgb(115,115,115)] hover:text-[rgb(163,163,163)] transition-colors cursor-pointer"
          >
            {logsOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            Logs
            {job.logs.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 rounded bg-[rgb(38,38,38)] text-[rgb(115,115,115)] text-[10px]">
                {job.logs.length}
              </span>
            )}
          </button>

          {/* Logs panel */}
          {logsOpen && (
            <div className="mt-2 bg-[rgb(12,12,12)] border border-[rgb(38,38,38)] rounded-lg max-h-48 overflow-y-auto p-3">
              {job.logs.length === 0 ? (
                <span className="text-[rgb(82,82,82)] text-xs font-mono">
                  Waiting for logs...
                </span>
              ) : (
                <div className="space-y-0.5">
                  {job.logs.map((line, i) => (
                    <div
                      key={i}
                      className={`text-[11px] font-mono leading-relaxed ${
                        line.includes("[ERROR]")
                          ? "text-rose-400/80"
                          : line.includes("[WARNING]")
                          ? "text-amber-400/70"
                          : "text-[rgb(163,163,163)]/70"
                      }`}
                    >
                      {line}
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
