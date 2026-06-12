"use client";

import { useState, useEffect, useCallback } from "react";
import { X, Folder, ChevronRight, ArrowUp } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  title: string;
  initialPath?: string;
}

interface BrowseResult {
  current: string;
  parent: string;
  dirs: string[];
  error?: string;
}

export default function FolderBrowser({ isOpen, onClose, onSelect, title, initialPath }: Props) {
  const [current, setCurrent] = useState("");
  const [parent, setParent] = useState("");
  const [dirs, setDirs] = useState<string[]>([]);
  const [pathInput, setPathInput] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const browse = useCallback(async (path: string) => {
    setLoading(true);
    setError("");
    try {
      const res = await apiFetch<BrowseResult>("/folders/browse", {
        method: "POST",
        body: JSON.stringify({ path: path.trim() }),
      });
      if (res.error) {
        setError(res.error);
        setPathInput(path.trim());
        return;
      }
      setCurrent(res.current);
      setParent(res.parent);
      setDirs(res.dirs);
      setPathInput(res.current);
    } catch (e) {
      console.error("Browse failed:", e);
      setError(e instanceof Error ? e.message : "Failed to browse folder");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleGo = () => {
    if (pathInput.trim()) browse(pathInput);
  };

  useEffect(() => {
    if (isOpen) {
      browse(initialPath || "");
    }
  }, [isOpen, browse, initialPath]);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="w-[520px] max-h-[70vh] bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
          <h3 className="text-sm font-semibold text-[rgb(245,245,245)]">{title}</h3>
          <button
            onClick={onClose}
            className="text-[rgb(115,115,115)] hover:text-white transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Path input — paste full path and press Enter or Go */}
        <div className="px-5 py-3 border-b border-[rgb(38,38,38)] space-y-2">
          <div className="flex items-center gap-2">
            {current && (
              <button
                onClick={() => browse(parent)}
                title="Parent folder"
                className="shrink-0 p-1.5 rounded hover:bg-[rgb(38,38,38)] text-[rgb(163,163,163)] transition-colors"
              >
                <ArrowUp size={16} />
              </button>
            )}
            <input
              type="text"
              value={pathInput}
              onChange={(e) => {
                setPathInput(e.target.value);
                setError("");
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleGo();
              }}
              placeholder="Paste or type folder path, e.g. D:\data\input"
              className="flex-1 min-w-0 px-3 py-1.5 text-xs font-mono bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-[rgb(229,229,229)] placeholder-[rgb(115,115,115)] focus:outline-none focus:border-[rgb(82,82,82)]"
            />
            <button
              onClick={handleGo}
              disabled={!pathInput.trim() || loading}
              className="shrink-0 px-3 py-1.5 text-xs rounded-lg bg-[rgb(38,38,38)] text-[rgb(212,212,212)] border border-[rgb(64,64,64)] hover:bg-[rgb(48,48,48)] disabled:opacity-40 transition-colors"
            >
              Go
            </button>
            <button
              onClick={() => current && onSelect(current)}
              disabled={!current}
              className="shrink-0 px-3 py-1.5 text-xs rounded-lg bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/25 disabled:opacity-40 transition-colors font-medium"
            >
              Select
            </button>
          </div>
          {error && (
            <p className="text-xs text-rose-400">{error}</p>
          )}
        </div>

        {/* Directory list */}
        <div className="flex-1 overflow-y-auto px-2 py-2">
          {loading ? (
            <div className="text-center py-8 text-sm text-[rgb(115,115,115)]">Loading...</div>
          ) : dirs.length === 0 ? (
            <div className="text-center py-8 text-sm text-[rgb(115,115,115)]">No subdirectories</div>
          ) : (
            dirs.map((name) => {
              const full = current ? `${current}${current.endsWith("\\") ? "" : "\\"}${name}` : name;
              return (
                <button
                  key={name}
                  onClick={() => browse(full)}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)] transition-colors text-left"
                >
                  <Folder size={16} className="text-amber-400 shrink-0" />
                  <span className="flex-1 truncate">{name}</span>
                  <ChevronRight size={14} className="text-[rgb(64,64,64)]" />
                </button>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
