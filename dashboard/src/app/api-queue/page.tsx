"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, RefreshCcw } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import { useConfig } from "@/hooks/useConfig";
import { apiFetch } from "@/lib/api";
import type {
  QueueJobItem,
  QueueJobListResponse,
  QueueStatus,
  ReviewStatus,
} from "@/types";

const PAGE_SIZE = 50;

type RetryResponse = {
  status: "queued" | "started";
  retry_type: "processing" | "plm_delivery";
};

const STATUS_OPTIONS: Array<{ label: string; value: "" | QueueStatus }> = [
  { label: "All", value: "" },
  { label: "Pending", value: "pending" },
  { label: "Running", value: "running" },
  { label: "Done", value: "done" },
  { label: "Failed", value: "failed" },
];

const queueStatusStyles: Record<QueueStatus, string> = {
  pending: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  running: "border-sky-500/30 bg-sky-500/10 text-sky-300",
  done: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  failed: "border-rose-500/30 bg-rose-500/10 text-rose-300",
};

const reviewStatusLabel: Record<ReviewStatus, string> = {
  not_required: "Auto Push",
  pending: "Waiting Review",
  approved: "Approved",
};

const reviewStatusStyles: Record<ReviewStatus, string> = {
  not_required: "border-[rgb(64,64,64)] bg-[rgb(38,38,38)] text-[rgb(163,163,163)]",
  pending: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  approved: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
};

const deliveryStatusLabel: Record<QueueJobItem["plm_delivery_status"], string> = {
  not_applicable: "N/A",
  pending: "Pending",
  uploaded: "Uploaded",
  complete: "Complete",
  failed: "Failed",
};

const deliveryStatusStyles: Record<QueueJobItem["plm_delivery_status"], string> = {
  not_applicable: "border-[rgb(64,64,64)] bg-[rgb(38,38,38)] text-[rgb(163,163,163)]",
  pending: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  uploaded: "border-sky-500/30 bg-sky-500/10 text-sky-300",
  complete: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  failed: "border-rose-500/30 bg-rose-500/10 text-rose-300",
};

const resultMethodLabel: Record<"O" | "N", string> = {
  O: "O",
  N: "N",
};

function resolveRefreshIntervalSeconds(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return 5;
  return parsed;
}

function formatDateTime(value: string | null): string {
  if (!value) return "\u2014";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function canRetryProcessing(item: QueueJobItem): boolean {
  return item.source === "api" && item.status === "failed";
}

function canRetryPlmDelivery(item: QueueJobItem): boolean {
  return (
    item.source === "api" &&
    item.status === "done" &&
    item.plm_delivery_status === "failed" &&
    (item.review_status === "approved" || item.review_status === "not_required")
  );
}

export default function ApiQueuePage() {
  const { config } = useConfig();
  const [items, setItems] = useState<QueueJobItem[]>([]);
  const [total, setTotal] = useState(0);
  const [counts, setCounts] = useState<Record<QueueStatus, number>>({
    pending: 0,
    running: 0,
    done: 0,
    failed: 0,
  });
  const [statusFilter, setStatusFilter] = useState<"" | QueueStatus>("");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [retryingJobId, setRetryingJobId] = useState<number | null>(null);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const refreshIntervalSeconds = useMemo(
    () => resolveRefreshIntervalSeconds(config?.current.API_QUEUE_REFRESH_INTERVAL),
    [config],
  );

  const loadQueue = useCallback(async (nextStatus: "" | QueueStatus, nextPage: number) => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        source: "api",
        limit: String(PAGE_SIZE),
        offset: String((nextPage - 1) * PAGE_SIZE),
      });
      if (nextStatus) params.set("status", nextStatus);
      const res = await apiFetch<QueueJobListResponse>(`/jobs/queue?${params.toString()}`);
      setItems(res.items);
      setTotal(res.total);
      setCounts(res.counts);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load API queue");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadQueue(statusFilter, page);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadQueue, page, statusFilter]);

  useEffect(() => {
    if (!autoRefresh) return;
    const timer = window.setInterval(() => {
      void loadQueue(statusFilter, page);
    }, refreshIntervalSeconds * 1000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, loadQueue, page, refreshIntervalSeconds, statusFilter]);

  const summaryCards = useMemo(
    () => [
      { label: "Pending", value: counts.pending, style: queueStatusStyles.pending },
      { label: "Running", value: counts.running, style: queueStatusStyles.running },
      { label: "Done", value: counts.done, style: queueStatusStyles.done },
      { label: "Failed", value: counts.failed, style: queueStatusStyles.failed },
    ],
    [counts],
  );

  const handleStatusChange = (value: "" | QueueStatus) => {
    setStatusFilter(value);
    setPage(1);
  };

  const handlePageChange = (nextPage: number) => {
    if (nextPage < 1 || nextPage > totalPages) return;
    setPage(nextPage);
  };

  const handleRetry = useCallback(
    async (item: QueueJobItem) => {
      setRetryingJobId(item.id);
      setError("");
      try {
        await apiFetch<RetryResponse>(`/jobs/queue/${item.id}/retry`, { method: "POST" });
        await loadQueue(statusFilter, page);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to retry queue item");
      } finally {
        setRetryingJobId(null);
      }
    },
    [loadQueue, page, statusFilter],
  );

  return (
    <div className="flex h-screen bg-[rgb(10,10,10)]">
      <Sidebar />

      <main className="flex-1 flex flex-col p-6 gap-5 overflow-hidden">
        <div className="shrink-0 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold text-[rgb(245,245,245)]">API Queue</h1>
            <p className="mt-1 text-sm text-[rgb(163,163,163)]">
              View API jobs with queue state, O/N result type, review state, PLM delivery progress, and retry actions.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <label className="inline-flex items-center gap-2 rounded-lg border border-[rgb(64,64,64)] px-3 py-2 text-sm text-[rgb(212,212,212)]">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="h-4 w-4 rounded border-[rgb(64,64,64)] bg-[rgb(38,38,38)]"
              />
              {`Auto refresh (${refreshIntervalSeconds}s)`}
            </label>
            <button
              onClick={() => void loadQueue(statusFilter, page)}
              className="inline-flex items-center gap-2 rounded-lg border border-[rgb(64,64,64)] px-3 py-2 text-sm text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)] transition-colors"
            >
              <RefreshCcw size={16} />
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="shrink-0 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {error}
          </div>
        )}

        <section className="shrink-0 grid grid-cols-2 xl:grid-cols-4 gap-4">
          {summaryCards.map((card) => (
            <div
              key={card.label}
              className="rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] p-4"
            >
              <div className="text-xs uppercase tracking-wider text-[rgb(115,115,115)]">{card.label}</div>
              <div className="mt-3 flex items-center justify-between">
                <div className="text-3xl font-semibold text-[rgb(245,245,245)]">{card.value}</div>
                <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs ${card.style}`}>
                  {card.label}
                </span>
              </div>
            </div>
          ))}
        </section>

        <section className="shrink-0 rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] p-5 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            {STATUS_OPTIONS.map((option) => {
              const active = statusFilter === option.value;
              return (
                <button
                  key={option.label}
                  onClick={() => handleStatusChange(option.value)}
                  className={`rounded-lg px-3 py-2 text-sm transition-colors ${
                    active
                      ? "bg-[rgb(245,245,245)] text-[rgb(10,10,10)]"
                      : "border border-[rgb(64,64,64)] text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)]"
                  }`}
                >
                  {option.label}
                </button>
              );
            })}
          </div>

          <div className="flex items-center justify-between text-sm text-[rgb(163,163,163)]">
            <span>Total: {total}</span>
            <span>
              Page {page} / {totalPages}
            </span>
          </div>
        </section>

        <section className="flex-1 min-h-0 flex flex-col rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] overflow-hidden">
          <div className="shrink-0 flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
            <div className="flex items-center gap-3">
              <h2 className="text-sm font-semibold text-[rgb(245,245,245)]">Queue Items</h2>
              {!loading && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-[rgb(38,38,38)] text-[rgb(163,163,163)]">
                  {items.length}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handlePageChange(page - 1)}
                disabled={page <= 1}
                className="p-1 rounded-lg hover:bg-[rgb(38,38,38)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-[rgb(163,163,163)]"
              >
                <ChevronLeft size={16} />
              </button>
              <button
                onClick={() => handlePageChange(page + 1)}
                disabled={page >= totalPages}
                className="p-1 rounded-lg hover:bg-[rgb(38,38,38)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-[rgb(163,163,163)]"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>

          <div className="flex-1 min-h-0 overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10 bg-[rgb(23,23,23)]">
                <tr className="border-b border-[rgb(38,38,38)] text-left">
                  <th className="px-5 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Drawing</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Queue</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Result</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Review</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">PLM Delivery</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Created</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Finished</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Actions</th>
                  <th className="px-4 py-3 text-xs font-medium uppercase tracking-wider text-[rgb(115,115,115)]">Detail</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={9} className="px-5 py-10 text-center text-sm text-[rgb(115,115,115)]">
                      Loading queue...
                    </td>
                  </tr>
                ) : items.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="px-5 py-10 text-center text-sm text-[rgb(115,115,115)]">
                      No API queue items found.
                    </td>
                  </tr>
                ) : (
                  items.map((item) => {
                    const retryProcessing = canRetryProcessing(item);
                    const retryDelivery = canRetryPlmDelivery(item);
                    const isRetrying = retryingJobId === item.id;

                    return (
                      <tr
                        key={item.id}
                        className="border-b border-[rgb(38,38,38)]/50 align-top hover:bg-[rgb(38,38,38)]/30 transition-colors"
                      >
                        <td className="px-5 py-3">
                          <div className="text-[rgb(229,229,229)] font-medium">
                            {item.drawing_no || "\u2014"}
                            {item.revision ? ` / ${item.revision}` : ""}
                          </div>
                          <div className="text-xs text-[rgb(115,115,115)] font-mono">{item.source_file}</div>
                          <div className="mt-1 text-xs text-[rgb(115,115,115)] font-mono">
                            doc={item.docnumber || "\u2014"} ws={item.work_seq || "\u2014"}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-col gap-2">
                            <span
                              className={`inline-flex w-fit rounded-full border px-2 py-0.5 text-xs ${queueStatusStyles[item.status]}`}
                            >
                              {item.status}
                            </span>
                            <span className="text-xs text-[rgb(115,115,115)]">retry={item.retry_count}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-col gap-2">
                            <span className="inline-flex w-fit rounded-full border border-[rgb(64,64,64)] bg-[rgb(38,38,38)] px-2 py-0.5 text-xs text-[rgb(212,212,212)]">
                              {item.result_method ? resultMethodLabel[item.result_method] : "Pending"}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-col gap-2">
                            <span
                              className={`inline-flex min-w-[124px] items-center justify-center whitespace-nowrap rounded-full border px-3 py-1 text-xs font-medium leading-none ${reviewStatusStyles[item.review_status]}`}
                            >
                              {reviewStatusLabel[item.review_status]}
                            </span>
                            {item.reviewed_at ? (
                              <span className="text-xs text-[rgb(115,115,115)]">
                                {formatDateTime(item.reviewed_at)}
                              </span>
                            ) : null}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-col gap-2">
                            <span
                              className={`inline-flex w-fit rounded-full border px-2 py-0.5 text-xs ${deliveryStatusStyles[item.plm_delivery_status]}`}
                            >
                              {deliveryStatusLabel[item.plm_delivery_status]}
                            </span>
                            {item.plm_delivery_error ? (
                              <div className="max-w-[240px] text-xs text-rose-300 break-all">
                                {item.plm_delivery_error}
                              </div>
                            ) : null}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-xs text-[rgb(163,163,163)]">
                          {formatDateTime(item.created_at)}
                        </td>
                        <td className="px-4 py-3 text-xs text-[rgb(163,163,163)]">
                          {formatDateTime(item.finished_at)}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-col gap-2">
                            {retryProcessing ? (
                              <button
                                onClick={() => void handleRetry(item)}
                                disabled={isRetrying}
                                className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-300 transition-colors hover:bg-amber-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                {isRetrying ? "Retrying..." : "Retry Processing"}
                              </button>
                            ) : null}
                            {retryDelivery ? (
                              <button
                                onClick={() => void handleRetry(item)}
                                disabled={isRetrying}
                                className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs text-sky-300 transition-colors hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                {isRetrying ? "Retrying..." : "Retry PLM Delivery"}
                              </button>
                            ) : null}
                            {!retryProcessing && !retryDelivery ? (
                              <span className="text-xs text-[rgb(115,115,115)]">
                                {item.review_status === "pending" ? "Waiting Review" : "\u2014"}
                              </span>
                            ) : null}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="space-y-1 text-xs text-[rgb(163,163,163)]">
                            <div className="font-mono break-all">src: {item.file_path}</div>
                            <div className="font-mono break-all">out: {item.result_path || "\u2014"}</div>
                            <div className="font-mono break-all">uploaded: {item.plm_uploaded_path || "\u2014"}</div>
                            {item.error_msg ? (
                              <div className="text-rose-300 break-all">err: {item.error_msg}</div>
                            ) : null}
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
