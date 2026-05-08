"use client";

import type { GpuMetrics } from "@/types";
import { Thermometer, Fan, Cpu, HardDrive, Clock, Zap } from "lucide-react";

interface Props {
  gpu: GpuMetrics;
}

function tempColor(t: number) {
  if (t < 50) return "text-emerald-400";
  if (t < 80) return "text-amber-400";
  return "text-rose-400";
}

function loadColor(l: number) {
  if (l < 30) return "bg-emerald-500";
  if (l < 70) return "bg-amber-500";
  return "bg-rose-500";
}

function loadTextColor(l: number) {
  if (l < 30) return "text-emerald-400";
  if (l < 70) return "text-amber-400";
  return "text-rose-400";
}

function ProgressBar({
  value,
  max,
  color,
}: {
  value: number;
  max: number;
  color: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-1.5 bg-[rgb(38,38,38)] rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function GPUWidget({ gpu }: Props) {
  const memPct = ((gpu.memoryUsed / gpu.memoryTotal) * 100).toFixed(1);

  return (
    <div className="bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] overflow-hidden">
      {/* GPU Header */}
      <div className="bg-[rgb(38,38,38)] px-4 py-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-[rgb(245,245,245)]">
          {gpu.name}
        </h3>
        <span className="text-xs px-2 py-0.5 rounded-full bg-[rgb(64,64,64)] text-[rgb(163,163,163)]">
          #{gpu.index}
        </span>
      </div>

      {/* Metrics */}
      <div className="p-4 space-y-4">
        {/* Row 1: Temperature + GPU Load */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-start gap-2.5">
            <Thermometer size={16} className="text-[rgb(115,115,115)] mt-0.5 shrink-0" />
            <div>
              <div className="text-xs text-[rgb(115,115,115)] mb-0.5">Temperature</div>
              <div className={`text-sm font-semibold ${tempColor(gpu.temperature)}`}>
                {gpu.temperature}°C
              </div>
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-1.5">
                <Cpu size={14} className="text-[rgb(115,115,115)]" />
                <span className="text-xs text-[rgb(115,115,115)]">GPU Load</span>
              </div>
              <span className={`text-xs font-semibold ${loadTextColor(gpu.gpuLoad)}`}>
                {gpu.gpuLoad}%
              </span>
            </div>
            <ProgressBar value={gpu.gpuLoad} max={100} color={loadColor(gpu.gpuLoad)} />
          </div>
        </div>

        <div className="border-t border-[rgb(38,38,38)]" />

        {/* Row 2: Fan Speed + Memory */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-start gap-2.5">
            <Fan size={16} className="text-[rgb(115,115,115)] mt-0.5 shrink-0" />
            <div>
              <div className="text-xs text-[rgb(115,115,115)] mb-0.5">Fan Speed</div>
              <div className="text-sm font-semibold text-blue-400">{gpu.fanSpeed}%</div>
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-1.5">
                <HardDrive size={14} className="text-[rgb(115,115,115)]" />
                <span className="text-xs text-[rgb(115,115,115)]">Memory</span>
              </div>
              <span className="text-xs font-semibold text-blue-400">{memPct}%</span>
            </div>
            <ProgressBar value={gpu.memoryUsed} max={gpu.memoryTotal} color="bg-blue-500" />
            <div className="text-xs text-[rgb(115,115,115)] mt-1">
              {gpu.memoryUsed} GB / {gpu.memoryTotal} GB
            </div>
          </div>
        </div>

        <div className="border-t border-[rgb(38,38,38)]" />

        {/* Row 3: Clock Speed + Power Draw */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-start gap-2.5">
            <Clock size={16} className="text-[rgb(115,115,115)] mt-0.5 shrink-0" />
            <div>
              <div className="text-xs text-[rgb(115,115,115)] mb-0.5">Clock Speed</div>
              <div className="text-sm font-semibold text-purple-400">
                {gpu.clockSpeed} MHz
              </div>
            </div>
          </div>
          <div className="flex items-start gap-2.5">
            <Zap size={16} className="text-[rgb(115,115,115)] mt-0.5 shrink-0" />
            <div>
              <div className="text-xs text-[rgb(115,115,115)] mb-0.5">Power Draw</div>
              <div className="text-sm font-semibold text-amber-400">
                {gpu.powerDraw}W{" "}
                <span className="text-[rgb(115,115,115)] font-normal">
                  / {gpu.powerLimit}W
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
