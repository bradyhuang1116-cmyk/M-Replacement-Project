"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Database, ScrollText, Settings, Power, RotateCcw } from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Datasets", href: "/datasets", icon: Database },
  { name: "Process Logs", href: "/process-logs", icon: ScrollText },
  { name: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [showShutdown, setShowShutdown] = useState(false);
  const [showRestart, setShowRestart] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [killDocker, setKillDocker] = useState(true);
  const [killWsl, setKillWsl] = useState(true);

  const handleRestart = async () => {
    setShowRestart(false);
    setRestarting(true);
    try {
      await fetch("http://localhost:8000/api/v1/system/restart", {
        method: "POST",
      });
    } catch {
      // Expected: backend exits before response completes
    }
    setTimeout(() => {
      setRestarting(false);
      window.location.reload();
    }, 5000);
  };

  const handleShutdown = async () => {
    setShowShutdown(false);
    try {
      await fetch("http://localhost:8000/api/v1/system/shutdown", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kill_docker: killDocker, kill_wsl: killWsl }),
      });
    } catch {
      // Expected: backend exits before response completes
    }
  };

  return (
    <>
      <aside className="flex flex-col w-60 bg-[rgb(23,23,23)] border-r border-[rgb(38,38,38)] h-full">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5">
          {/*<Image*/}
          {/*  src="/logo.png"*/}
          {/*  alt="Logo"*/}
          {/*  width={32}*/}
          {/*  height={32}*/}
          {/*  className="rounded-lg"*/}
          {/*  style={{ width: "auto", height: "auto" }}*/}
          {/*/>*/}
          <div className="leading-tight">
            <span className="text-[rgb(245,245,245)] font-bold text-sm tracking-wide">
              Blueprint
            </span>
            <br />
            <span className="text-[rgb(163,163,163)] text-xs tracking-wide">
              Replacement Service
            </span>
          </div>
        </div>

        {/* Divider */}
        <div className="mx-4 border-t border-[rgb(38,38,38)]" />

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navigation.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm transition-colors ${
                  active
                    ? "bg-[rgb(38,38,38)] text-white"
                    : "text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)]"
                }`}
              >
                <item.icon size={18} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-3 pb-3 space-y-2">
          <button
            onClick={() => setShowRestart(true)}
            disabled={restarting}
            className="flex items-center gap-3 w-full px-4 py-2.5 rounded-lg text-sm text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)] transition-colors disabled:opacity-50"
          >
            <RotateCcw size={18} className={restarting ? "animate-spin" : ""} />
            {restarting ? "Restarting..." : "Restart Backend"}
          </button>
          <button
            onClick={() => setShowShutdown(true)}
            className="flex items-center gap-3 w-full px-4 py-2.5 rounded-lg text-sm text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)] transition-colors"
          >
            <Power size={18} />
            Shutdown
          </button>
          <div className="px-4 text-xs text-[rgb(115,115,115)]">v2.0.0</div>
        </div>
      </aside>

      {/* Restart confirmation modal */}
      {showRestart && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[rgb(23,23,23)] border border-[rgb(38,38,38)] rounded-xl p-6 max-w-sm mx-4 space-y-5">
            <div className="flex items-center gap-3">
              <RotateCcw size={20} className="text-blue-400 shrink-0" />
              <h3 className="text-[rgb(245,245,245)] font-semibold text-base">
                Restart Backend
              </h3>
            </div>
            <p className="text-sm text-[rgb(163,163,163)] leading-relaxed">
              The backend server will be restarted. The frontend will reload after the restart completes.
            </p>
            <div className="flex gap-3 justify-end pt-1">
              <button
                onClick={() => setShowRestart(false)}
                className="px-4 py-2 rounded-lg text-sm text-[rgb(163,163,163)] hover:bg-[rgb(38,38,38)] transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleRestart}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/25 transition-colors"
              >
                Restart
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Shutdown confirmation modal */}
      {showShutdown && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-[rgb(23,23,23)] border border-[rgb(38,38,38)] rounded-xl p-6 max-w-sm mx-4 space-y-5">
            <h3 className="text-[rgb(245,245,245)] font-semibold text-base">
              Shutdown System
            </h3>

            <div className="space-y-3">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={killDocker}
                  onChange={(e) => setKillDocker(e.target.checked)}
                  className="w-4 h-4 rounded border-[rgb(64,64,64)] bg-[rgb(38,38,38)] text-blue-500 focus:ring-blue-500/30 focus:ring-offset-0"
                />
                <span className="text-sm text-[rgb(212,212,212)]">
                  Docker Desktop
                </span>
              </label>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={killWsl}
                  onChange={(e) => setKillWsl(e.target.checked)}
                  className="w-4 h-4 rounded border-[rgb(64,64,64)] bg-[rgb(38,38,38)] text-blue-500 focus:ring-blue-500/30 focus:ring-offset-0"
                />
                <span className="text-sm text-[rgb(212,212,212)]">WSL</span>
              </label>
            </div>

            <div className="flex gap-3 justify-end pt-1">
              <button
                onClick={() => setShowShutdown(false)}
                className="px-4 py-2 rounded-lg text-sm text-[rgb(163,163,163)] hover:bg-[rgb(38,38,38)] transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleShutdown}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-rose-500/15 text-rose-400 border border-rose-500/30 hover:bg-rose-500/25 transition-colors"
              >
                Shutdown
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
