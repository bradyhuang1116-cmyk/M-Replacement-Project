"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { apiFetch, apiFetchBlob } from "@/lib/api";
import type { PaginatedLogs, ProcessLogItem, ProcessLogStatus, ProcessMode } from "@/types";
import DatePicker from "@/components/DatePicker";
import { Download, RefreshCcw, Search } from "lucide-react";

interface Filters {
  drawing_no: string;
  revision: string;
  status: "" | ProcessLogStatus;
  mode: "" | ProcessMode;
  from: string;
  to: string;
}

const DEFAULT_FILTERS: Filters = {
  drawing_no: "",
  revision: "",
  status: "",
  mode: "",
  from: "",
  to: "",
};

function buildQuery(filters: Filters): string {
  const params = new URLSearchParams();
  if (filters.drawing_no) params.set("drawing_no", filters.drawing_no);
  if (filters.revision) params.set("revision", filters.revision);
  if (filters.status) params.set("status", filters.status);
  if (filters.mode) params.set("mode", filters.mode);
  if (filters.from) params.set("from", filters.from);
  if (filters.to) params.set("to", filters.to);
  params.set("limit", "200");
  return params.toString();
}

export default function ProcessLogsPage() {
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [items, setItems] = useState<ProcessLogItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");

  const loadItems = useCallback(async (nextFilters: Filters) => {
    setLoading(true);
    setError("");
    try {
      const query = buildQuery(nextFilters);
      const res = await apiFetch<PaginatedLogs>(`/logs?${query}`);
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load logs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadItems(DEFAULT_FILTERS);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadItems]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (item) =>
        item.drawing_no.toLowerCase().includes(q) ||
        item.filename.toLowerCase().includes(q)
    );
  }, [items, search]);

  const downloadCsv = async () => {
    try {
      const query = buildQuery(filters).replace(/(^|&)limit=200(&|$)/, "$1").replace(/^&|&$/g, "");
      const blob = await apiFetchBlob(`/logs/export${query ? `?${query}` : ""}`);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "process_logs.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to export CSV");
    }
  };

  const inputClassName =
    "w-full px-3 py-2 text-sm bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-[rgb(229,229,229)] placeholder-[rgb(115,115,115)] focus:outline-none focus:border-[rgb(82,82,82)] transition-colors";

  return (
    <div className="flex h-screen bg-[rgb(10,10,10)]">
      <Sidebar />

      <main className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold text-[rgb(245,245,245)]">Process Logs</h1>
            <p className="mt-1 text-sm text-[rgb(163,163,163)]">
              Query automatic processing records with filters for drawing no., revision, date, mode (O/N), and CSV export.
            </p>
          </div>
        </div>

        {error && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {error}
          </div>
        )}

        <section className="rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] p-5 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-5 gap-4">
            <input
              value={filters.drawing_no}
              onChange={(e) => setFilters((prev) => ({ ...prev, drawing_no: e.target.value }))}
              placeholder="Drawing No."
              className={inputClassName}
            />
            <input
              value={filters.revision}
              onChange={(e) => setFilters((prev) => ({ ...prev, revision: e.target.value }))}
              placeholder="Revision"
              className={inputClassName}
            />
            <select
              value={filters.status}
              onChange={(e) => setFilters((prev) => ({ ...prev, status: e.target.value as Filters["status"] }))}
              className={inputClassName}
            >
              <option value="">All Status</option>
              <option value="success">Success</option>
              <option value="failed">Failed</option>
            </select>
            <select
              value={filters.mode}
              onChange={(e) => setFilters((prev) => ({ ...prev, mode: e.target.value as Filters["mode"] }))}
              className={inputClassName}
            >
              <option value="">All Mode</option>
              <option value="N">N · Vector PDF</option>
              <option value="O">O · VLM OCR</option>
            </select>
            <div className="relative">
              <Search
                size={14}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[rgb(115,115,115)]"
              />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Quick search loaded rows..."
                className="w-full pl-8 pr-3 py-2 text-sm bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-[rgb(229,229,229)] placeholder-[rgb(115,115,115)] focus:outline-none focus:border-[rgb(82,82,82)] transition-colors"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <DatePicker
              label="From Date"
              value={filters.from}
              onChange={(v) => setFilters((prev) => ({ ...prev, from: v }))}
            />
            <DatePicker
              label="To Date"
              value={filters.to}
              onChange={(v) => setFilters((prev) => ({ ...prev, to: v }))}
            />
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => void loadItems(filters)}
              className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 transition-colors"
            >
              Apply Filters
            </button>
            <button
              onClick={() => {
                setFilters(DEFAULT_FILTERS);
                setSearch("");
                void loadItems(DEFAULT_FILTERS);
              }}
              className="px-4 py-2 rounded-lg border border-[rgb(64,64,64)] text-sm text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)] transition-colors"
            >
              Clear
            </button>
            <div className="w-px bg-[rgb(64,64,64)]" />
            <button
              onClick={() => void loadItems(filters)}
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-[rgb(64,64,64)] bg-[rgb(23,23,23)] text-sm text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)] transition-colors"
            >
              <RefreshCcw size={16} />
              Refresh
            </button>
            <button
              onClick={() => void downloadCsv()}
              className="ml-auto inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-blue-500/30 bg-blue-500/15 text-sm text-blue-300 hover:bg-blue-500/25 transition-colors"
            >
              <Download size={16} />
              Export CSV
            </button>
          </div>
        </section>

        <section className="rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
            <div className="flex items-center gap-3">
              <h2 className="text-sm font-semibold text-[rgb(245,245,245)]">Log Records</h2>
              <span className="text-xs px-2 py-0.5 rounded-full bg-[rgb(38,38,38)] text-[rgb(163,163,163)]">
                {filtered.length}/{total}
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[rgb(38,38,38)] text-left">
                  <th className="px-5 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Drawing</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Status</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Mode</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Revision</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Processed At</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={5} className="px-5 py-10 text-center text-sm text-[rgb(115,115,115)]">
                      Loading...
                    </td>
                  </tr>
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-5 py-10 text-center text-sm text-[rgb(115,115,115)]">
                      No logs match current filters.
                    </td>
                  </tr>
                ) : (
                  filtered.map((item) => (
                    <tr
                      key={item.id}
                      className="border-b border-[rgb(38,38,38)]/50 hover:bg-[rgb(38,38,38)]/30 transition-colors"
                    >
                      <td className="px-5 py-3">
                        <div className="text-[rgb(229,229,229)] font-medium">{item.drawing_no || "—"}</div>
                        <div className="text-xs text-[rgb(115,115,115)] font-mono">{item.filename}</div>
                      </td>
                      <td className="px-4 py-3 text-xs">
                        <span
                          className={`inline-flex px-2 py-0.5 rounded-full border ${
                            item.status === "success"
                              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                              : "border-rose-500/30 bg-rose-500/10 text-rose-300"
                          }`}
                        >
                          {item.status === "success" ? "Success" : "Failed"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-xs text-[rgb(163,163,163)]">
                        {item.ocr_flag === "N" ? "Vector PDF" : "VLM OCR"}
                      </td>
                      <td className="px-4 py-3 text-xs text-[rgb(163,163,163)]">{item.revision || "—"}</td>
                      <td className="px-4 py-3 text-xs text-[rgb(163,163,163)]">
                        {new Date(item.process_date).toLocaleString()}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
