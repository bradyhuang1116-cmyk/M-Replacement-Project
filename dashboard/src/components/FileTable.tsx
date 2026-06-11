"use client";

import { useState, useMemo } from "react";
import type { FileItem } from "@/types";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";

interface Props {
  files: FileItem[];
}

const PAGE_SIZE = 10;

const statusConfig = {
  completed: { label: "Completed", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" },
  processing: { label: "Processing", color: "text-amber-400 bg-amber-500/10 border-amber-500/20" },
  pending: { label: "Pending", color: "text-[rgb(115,115,115)] bg-[rgb(38,38,38)] border-[rgb(64,64,64)]" },
  failed: { label: "Failed", color: "text-rose-400 bg-rose-500/10 border-rose-500/20" },
};

export default function FileTable({ files }: Props) {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    if (!search.trim()) return files;
    const q = search.toLowerCase();
    return files.filter(
      (f) =>
        f.filename.toLowerCase().includes(q) ||
        f.replacedFilename.toLowerCase().includes(q)
    );
  }, [files, search]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageFiles = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div className="bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-[rgb(245,245,245)]">
            File Queue
          </h3>
          <span className="text-xs px-2 py-0.5 rounded-full bg-[rgb(38,38,38)] text-[rgb(163,163,163)]">
            {files.length}
          </span>
        </div>
        <div className="relative">
          <Search
            size={14}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[rgb(115,115,115)]"
          />
          <input
            type="text"
            placeholder="Search files..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="w-52 pl-8 pr-3 py-1.5 text-xs bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-[rgb(229,229,229)] placeholder-[rgb(115,115,115)] focus:outline-none focus:border-[rgb(82,82,82)] transition-colors"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[rgb(38,38,38)] text-left">
              <th className="px-5 py-3 text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                Filename
              </th>
              <th className="px-4 py-3 text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                Status
              </th>
              <th className="px-4 py-3 text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                Duration
              </th>
              <th className="px-4 py-3 text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                Method
              </th>
              <th className="px-4 py-3 text-xs font-medium text-[rgb(115,115,115)] uppercase tracking-wider">
                Output
              </th>
            </tr>
          </thead>
          <tbody>
            {pageFiles.map((file) => {
              const st = statusConfig[file.status];
              const isProcessing = file.status === "processing";
              return (
                <tr
                  key={file.id}
                  className={`border-b border-[rgb(38,38,38)]/50 transition-colors hover:bg-[rgb(38,38,38)]/30 ${
                    isProcessing ? "bg-amber-500/[0.03]" : ""
                  }`}
                >
                  <td className="px-5 py-3 text-[rgb(229,229,229)] font-mono text-xs">
                    {file.filename}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${st.color}`}
                    >
                      {isProcessing && (
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-amber-400" />
                        </span>
                      )}
                      {st.label}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-[rgb(163,163,163)]">
                    {file.status === "completed" ? file.duration : "—"}
                  </td>
                  <td className="px-4 py-3 text-xs text-[rgb(163,163,163)]">
                    {file.status === "failed" && file.error ? (
                      <span
                        className="text-rose-400 block truncate max-w-[200px]"
                        title={file.error}
                      >
                        {file.error}
                      </span>
                    ) : file.status === "completed" && file.method ? (
                      <>
                        {file.method}
                        <span className="text-[rgb(82,82,82)]"> · </span>
                        <span className="text-[rgb(115,115,115)]">
                          /{file.method === "PDF Replacement" ? "PDF_Replacement" : "VLMOCR"}
                        </span>
                      </>
                    ) : "—"}
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-[rgb(163,163,163)]">
                    {file.status === "completed" ? file.replacedFilename : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-5 py-3 border-t border-[rgb(38,38,38)]">
          <span className="text-xs text-[rgb(115,115,115)]">
            Page {page} of {totalPages}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-1.5 rounded-lg hover:bg-[rgb(38,38,38)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-[rgb(163,163,163)]"
            >
              <ChevronLeft size={16} />
            </button>
            {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
              <button
                key={p}
                onClick={() => setPage(p)}
                className={`min-w-[28px] h-7 rounded-lg text-xs transition-colors ${
                  p === page
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "text-[rgb(163,163,163)] hover:bg-[rgb(38,38,38)]"
                }`}
              >
                {p}
              </button>
            ))}
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="p-1.5 rounded-lg hover:bg-[rgb(38,38,38)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-[rgb(163,163,163)]"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
