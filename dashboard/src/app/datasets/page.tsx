"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import FolderBrowser from "@/components/FolderBrowser";
import { apiFetch } from "@/lib/api";
import type { ConfigResponse } from "@/types";
import { FolderOpen, FileText, Play } from "lucide-react";

interface ScannedFile {
  name: string;
  size: number;
  ext: string;
}

const PREFIXES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export default function DatasetsPage() {
  const router = useRouter();
  const [inputDir, setInputDir] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [selectedPrefixes, setSelectedPrefixes] = useState<string[]>(["Y"]);
  const [browseTarget, setBrowseTarget] = useState<"input" | "output" | null>(null);
  const [scannedFiles, setScannedFiles] = useState<ScannedFile[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [scanning, setScanning] = useState(false);
  const [starting, setStarting] = useState(false);

  const scanFiles = useCallback(async (path: string) => {
    setScanning(true);
    try {
      const res = await apiFetch<{ files: ScannedFile[]; total: number }>(
        `/folders/scan?path=${encodeURIComponent(path)}`
      );
      setScannedFiles(res.files);
      setSelectedFiles(new Set(res.files.map((f) => f.name)));
    } catch {
      setScannedFiles([]);
      setSelectedFiles(new Set());
    } finally {
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const data = await apiFetch<ConfigResponse>("/config");
        const inbox = String(data.current.WATCH_INBOX_DIR ?? "").trim();
        const output = String(data.current.WATCH_OUTPUT_DIR ?? "").trim();
        if (inbox) setInputDir(inbox);
        if (output) setOutputDir(output);
      } catch (e) {
        console.error("Failed to load watch folder config:", e);
      }
    })();
  }, []);

  useEffect(() => {
    if (inputDir) scanFiles(inputDir);
    else {
      setScannedFiles([]);
      setSelectedFiles(new Set());
    }
  }, [inputDir, scanFiles]);

  const allSelected = scannedFiles.length > 0 && selectedFiles.size === scannedFiles.length;

  const toggleSelectAll = () => {
    if (allSelected) {
      setSelectedFiles(new Set());
    } else {
      setSelectedFiles(new Set(scannedFiles.map((f) => f.name)));
    }
  };

  const toggleFile = (name: string) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const handleStart = async () => {
    if (!inputDir || !outputDir || selectedFiles.size === 0) return;
    setStarting(true);
    try {
      await apiFetch("/jobs/start", {
        method: "POST",
        body: JSON.stringify({
          input_dir: inputDir,
          output_dir: outputDir,
          prefixes: selectedPrefixes,
          selected_files: Array.from(selectedFiles),
        }),
      });
      router.push("/");
    } catch (e) {
      console.error("Failed to start:", e);
    } finally {
      setStarting(false);
    }
  };

  const togglePrefix = (p: string) => {
    setSelectedPrefixes((prev) =>
      prev.includes(p) ? prev.filter((x) => x !== p) : [...prev, p]
    );
  };

  const allPrefixesSelected = selectedPrefixes.length === PREFIXES.length;

  const toggleAllPrefixes = () => {
    if (allPrefixesSelected) {
      setSelectedPrefixes([]);
    } else {
      setSelectedPrefixes([...PREFIXES]);
    }
  };

  const prefixToggleLabel = allPrefixesSelected
    ? "Deselect All"
    : selectedPrefixes.length === 0
      ? "Select All"
      : `Select All (${selectedPrefixes.length}/${PREFIXES.length})`;

  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  const handleFolderSelect = (path: string) => {
    if (browseTarget === "input") {
      setInputDir(path);
    } else {
      setOutputDir(path);
    }
    setBrowseTarget(null);
  };

  const getBrowseInitialPath = (): string => {
    if (browseTarget === "input") return inputDir;
    if (browseTarget === "output") return outputDir;
    return "";
  };

  const canStart =
    !!inputDir &&
    !!outputDir &&
    selectedPrefixes.length > 0 &&
    selectedFiles.size > 0 &&
    !starting;

  const inputClassName =
    "flex-1 px-3 py-2 text-sm bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-[rgb(229,229,229)] placeholder-[rgb(115,115,115)] focus:outline-none focus:border-[rgb(82,82,82)]";

  return (
    <div className="flex h-screen bg-[rgb(10,10,10)]">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Scrollable upper area: title + config cards */}
        <div className="flex-1 min-h-0 flex flex-col p-6 gap-5 overflow-hidden">
          <h1 className="shrink-0 text-lg font-semibold text-[rgb(245,245,245)]">Datasets</h1>

          <div className="shrink-0 grid grid-cols-2 gap-5">
            <div className="bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] p-5">
              <label className="text-sm font-medium text-[rgb(212,212,212)] mb-2 block">
                Input Folder
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputDir}
                  onChange={(e) => setInputDir(e.target.value)}
                  placeholder="Paste path or browse..."
                  className={inputClassName}
                />
                <button
                  onClick={() => setBrowseTarget("input")}
                  className="px-3 py-2 rounded-lg bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] text-[rgb(163,163,163)] hover:bg-[rgb(48,48,48)] transition-colors"
                >
                  <FolderOpen size={16} />
                </button>
              </div>
            </div>

            <div className="bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] p-5">
              <label className="text-sm font-medium text-[rgb(212,212,212)] mb-2 block">
                Output Folder
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                  placeholder="Paste path or browse..."
                  className={inputClassName}
                />
                <button
                  onClick={() => setBrowseTarget("output")}
                  className="px-3 py-2 rounded-lg bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] text-[rgb(163,163,163)] hover:bg-[rgb(48,48,48)] transition-colors"
                >
                  <FolderOpen size={16} />
                </button>
              </div>
            </div>
          </div>

          <div className="shrink-0 bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] p-5">
            <label className="text-sm font-medium text-[rgb(212,212,212)] mb-3 block">
              Detection Prefixes
            </label>
            <div className="flex flex-wrap items-center gap-1.5">
              {PREFIXES.map((p) => (
                <button
                  key={p}
                  onClick={() => togglePrefix(p)}
                  className={`w-8 h-8 rounded-lg text-xs font-medium transition-colors ${
                    selectedPrefixes.includes(p)
                      ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                      : "bg-[rgb(38,38,38)] text-[rgb(115,115,115)] border border-[rgb(64,64,64)] hover:bg-[rgb(48,48,48)]"
                  }`}
                >
                  {p}
                </button>
              ))}
              <button
                onClick={toggleAllPrefixes}
                className={`h-8 px-3 rounded-lg text-xs font-medium transition-colors ${
                  allPrefixesSelected
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "bg-[rgb(38,38,38)] text-[rgb(163,163,163)] border border-[rgb(64,64,64)] hover:bg-[rgb(48,48,48)]"
                }`}
              >
                {prefixToggleLabel}
              </button>
            </div>
          </div>

          {/* File Preview — flex-1 fills remaining viewport, internal scroll */}
          <section className="flex-1 min-h-0 flex flex-col bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] overflow-hidden">
            <div className="shrink-0 flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold text-[rgb(245,245,245)]">File Preview</h2>
                <span className="text-xs px-2 py-0.5 rounded-full bg-[rgb(38,38,38)] text-[rgb(163,163,163)]">
                  {selectedFiles.size}/{scannedFiles.length}
                </span>
              </div>
              {scannedFiles.length > 0 && (
                <button
                  onClick={toggleSelectAll}
                  className="text-xs px-3 py-1 rounded-lg bg-[rgb(38,38,38)] text-[rgb(163,163,163)] hover:bg-[rgb(48,48,48)] border border-[rgb(64,64,64)] transition-colors"
                >
                  {allSelected ? "Deselect All" : "Select All"}
                </button>
              )}
            </div>

            <div className="flex-1 min-h-0 overflow-auto">
              {!inputDir ? (
                <div className="flex items-center justify-center h-full text-sm text-[rgb(115,115,115)]">
                  Select an input folder to preview files
                </div>
              ) : scanning ? (
                <div className="flex items-center justify-center h-full text-sm text-[rgb(115,115,115)]">
                  Scanning...
                </div>
              ) : scannedFiles.length === 0 ? (
                <div className="flex items-center justify-center h-full text-sm text-[rgb(115,115,115)]">
                  No supported files found
                </div>
              ) : (
                <table className="w-full text-sm">
                  <thead className="sticky top-0 z-10 bg-[rgb(23,23,23)]">
                    <tr className="border-b border-[rgb(38,38,38)]">
                      <th className="px-3 py-3 w-10">
                        <input
                          type="checkbox"
                          checked={allSelected}
                          onChange={toggleSelectAll}
                          className="w-3.5 h-3.5 rounded border-[rgb(64,64,64)] bg-[rgb(38,38,38)] text-blue-500 focus:ring-blue-500/30 focus:ring-offset-0"
                        />
                      </th>
                      <th className="px-3 py-3 text-left text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                        Filename
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                        Size
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {scannedFiles.map((f) => (
                      <tr
                        key={f.name}
                        className={`border-b border-[rgb(38,38,38)]/50 hover:bg-[rgb(38,38,38)]/30 cursor-pointer transition-colors ${
                          !selectedFiles.has(f.name) ? "opacity-40" : ""
                        }`}
                        onClick={() => toggleFile(f.name)}
                      >
                        <td className="px-3 py-3 w-10">
                          <input
                            type="checkbox"
                            checked={selectedFiles.has(f.name)}
                            onChange={() => toggleFile(f.name)}
                            onClick={(e) => e.stopPropagation()}
                            className="w-3.5 h-3.5 rounded border-[rgb(64,64,64)] bg-[rgb(38,38,38)] text-blue-500 focus:ring-blue-500/30 focus:ring-offset-0"
                          />
                        </td>
                        <td className="px-3 py-3 text-xs font-mono text-[rgb(229,229,229)]">
                          <span className="inline-flex items-center gap-2">
                            <FileText size={14} className="text-[rgb(115,115,115)] shrink-0" />
                            {f.name}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-xs text-[rgb(163,163,163)] uppercase">
                          {f.ext.replace(".", "")}
                        </td>
                        <td className="px-4 py-3 text-xs text-[rgb(163,163,163)] text-right">
                          {formatSize(f.size)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </section>
        </div>

        {/* Sticky bottom bar — same as Settings */}
        <div className="shrink-0 border-t border-[rgb(38,38,38)] bg-[rgb(10,10,10)] flex items-center justify-end gap-3 px-6 h-16">
          <button
            onClick={handleStart}
            disabled={!canStart}
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-lg"
          >
            <Play size={18} fill="currentColor" />
            {starting ? "Starting..." : "Start Replacing"}
          </button>
        </div>
      </main>

      <FolderBrowser
        isOpen={browseTarget !== null}
        onClose={() => setBrowseTarget(null)}
        title={browseTarget === "input" ? "Select Input Folder" : "Select Output Folder"}
        initialPath={browseTarget ? getBrowseInitialPath() : ""}
        onSelect={handleFolderSelect}
      />
    </div>
  );
}
