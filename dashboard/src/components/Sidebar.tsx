"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Database, ScrollText, ListOrdered, ClipboardCheck, Settings, Power, RotateCcw } from "lucide-react";
import ConfirmDialog from "@/components/ConfirmDialog";
import { getApiBaseUrl } from "@/lib/api";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Datasets", href: "/datasets", icon: Database },
  { name: "API Queue", href: "/api-queue", icon: ListOrdered },
  { name: "Pending Review", href: "/pending-review", icon: ClipboardCheck },
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
      await fetch(`${getApiBaseUrl()}/system/restart`, {
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
      await fetch(`${getApiBaseUrl()}/system/shutdown`, {
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

      <ConfirmDialog
        open={showRestart}
        onClose={() => setShowRestart(false)}
        onConfirm={handleRestart}
        title="Restart Backend"
        description="Backend will restart. Page reloads when it's back."
        icon={<RotateCcw size={20} className="text-blue-400 shrink-0" />}
        confirmLabel="Restart"
        variant="default"
      />

      <ConfirmDialog
        open={showShutdown}
        onClose={() => setShowShutdown(false)}
        onConfirm={handleShutdown}
        title="Shutdown System"
        description="Stops the backend. Uncheck items you want to keep running."
        icon={<Power size={20} className="text-rose-400 shrink-0" />}
        confirmLabel="Shutdown"
        variant="danger"
      >
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={killDocker}
            onChange={(e) => setKillDocker(e.target.checked)}
            className="w-4 h-4 rounded border-[rgb(64,64,64)] bg-[rgb(38,38,38)] text-blue-500 focus:ring-blue-500/30 focus:ring-offset-0"
          />
          <span className="text-sm text-[rgb(212,212,212)]">Docker Desktop</span>
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
      </ConfirmDialog>
    </>
  );
}
