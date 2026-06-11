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
  const [loading, setLoading] = useState(false);

  const browse = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const res = await apiFetch<BrowseResult>("/folders/browse", {
        method: "POST",
        body: JSON.stringify({ path }),
      });
      setCurrent(res.current);
      setParent(res.parent);
      setDirs(res.dirs);
    } catch (e) {
      console.error("Browse failed:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      browse(initialPath || "");
    }
  }, [isOpen, browse, initialPath]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-[520px] max-h-[70vh] bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
          <h3 className="text-sm font-semibold text-[rgb(245,245,245)]">{title}</h3>
          <button onClick={onClose} className="text-[rgb(115,115,115)] hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* Current path */}
        <div className="px-5 py-2 border-b border-[rgb(38,38,38)] flex items-center gap-2">
          {current && (
            <button
              onClick={() => browse(parent)}
              className="p-1 rounded hover:bg-[rgb(38,38,38)] text-[rgb(163,163,163)] transition-colors"
            >
              <ArrowUp size={16} />
            </button>
          )}
          <span className="text-xs text-[rgb(163,163,163)] font-mono truncate flex-1">
            {current || "Select a drive"}
          </span>
          {current && (
            <button
              onClick={() => onSelect(current)}
              className="px-3 py-1 text-xs rounded-lg bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/25 transition-colors font-medium"
            >
              Select
            </button>
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
