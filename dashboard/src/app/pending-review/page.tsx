"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, ChevronLeft, ChevronRight, ExternalLink, RefreshCcw } from "lucide-react";
import ConfirmDialog from "@/components/ConfirmDialog";
import Sidebar from "@/components/Sidebar";
import { apiFetch, apiFetchBlob } from "@/lib/api";
import type { QueueJobItem, ReviewJobListResponse } from "@/types";

const PAGE_SIZE = 24;
const DEFAULT_MAGNIFIER_SIZE = 220;
const DEFAULT_MAGNIFIER_ZOOM = 2.5;

function formatDateTime(value: string | null): string {
  if (!value) return "\u2014";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function getErrorDetail(error: unknown, fallback: string): string {
  if (!(error instanceof Error)) return fallback;

  const message = error.message || fallback;
  const jsonStart = message.indexOf("{");
  if (jsonStart >= 0) {
    try {
      const parsed = JSON.parse(message.slice(jsonStart)) as { detail?: unknown };
      if (typeof parsed.detail === "string" && parsed.detail.trim()) {
        return parsed.detail;
      }
    } catch {
      // Fall through to plain-text handling.
    }
  }

  const detailMatch = message.match(/detail[:=]\s*(.+)$/i);
  if (detailMatch?.[1]) return detailMatch[1].trim();

  const apiPrefix = message.match(/^API error \d+:\s*(.+)$/i);
  if (apiPrefix?.[1]) return apiPrefix[1].trim();

  return message;
}

export default function PendingReviewPage() {
  const [items, setItems] = useState<QueueJobItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [reviewRoot, setReviewRoot] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [approvingId, setApprovingId] = useState<number | null>(null);
  const [manualProcessingId, setManualProcessingId] = useState<number | null>(null);
  const [actionMessage, setActionMessage] = useState("");
  const [remoteAlertMessage, setRemoteAlertMessage] = useState("");
  const [remoteAlertOpen, setRemoteAlertOpen] = useState(false);
  const [magnifierSize, setMagnifierSize] = useState(DEFAULT_MAGNIFIER_SIZE);
  const [magnifierZoom, setMagnifierZoom] = useState(DEFAULT_MAGNIFIER_ZOOM);
  const [magnifier, setMagnifier] = useState({
    visible: false,
    lensX: 0,
    lensY: 0,
    bgX: 0,
    bgY: 0,
    bgWidth: 0,
    bgHeight: 0,
  });
  const previewFrameRef = useRef<HTMLDivElement | null>(null);
  const previewImageRef = useRef<HTMLImageElement | null>(null);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const selectedItem = useMemo(
    () => items.find((item) => item.id === selectedId) ?? null,
    [items, selectedId],
  );

  const loadItems = useCallback(async (nextPage: number) => {
    setLoading(true);
    setPageError("");
    setPreviewError("");
    setActionMessage("");
    setRemoteAlertOpen(false);
    try {
      const params = new URLSearchParams({
        limit: String(PAGE_SIZE),
        offset: String((nextPage - 1) * PAGE_SIZE),
      });
      const res = await apiFetch<ReviewJobListResponse>(`/jobs/review?${params.toString()}`);
      setItems(res.items);
      setTotal(res.total);
      setReviewRoot(res.review_root);
      setSelectedId((prev) => {
        if (res.items.length === 0) return null;
        if (prev && res.items.some((item) => item.id === prev)) return prev;
        return res.items[0].id;
      });
    } catch (e) {
      setPageError(getErrorDetail(e, "Failed to load pending review jobs"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadItems(page);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadItems, page]);

  useEffect(() => {
    let revokedUrl = "";
    let active = true;

    const loadPreview = async () => {
      if (!selectedItem) {
        setPreviewUrl("");
        setMagnifier((prev) => ({ ...prev, visible: false }));
        return;
      }

      setPreviewLoading(true);
      try {
        const blob = await apiFetchBlob(`/jobs/review/${selectedItem.id}/preview`);
        if (!active) return;
        revokedUrl = URL.createObjectURL(blob);
        setPreviewError("");
        setPreviewUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return revokedUrl;
        });
        setMagnifier((prev) => ({ ...prev, visible: false }));
      } catch (e) {
        if (!active) return;
        setPreviewUrl("");
        setPreviewError(getErrorDetail(e, "Failed to load preview"));
        setMagnifier((prev) => ({ ...prev, visible: false }));
      } finally {
        if (active) setPreviewLoading(false);
      }
    };

    void loadPreview();

    return () => {
      active = false;
      if (revokedUrl) URL.revokeObjectURL(revokedUrl);
    };
  }, [selectedItem]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handlePageChange = (nextPage: number) => {
    if (nextPage < 1 || nextPage > totalPages) return;
    setPage(nextPage);
  };

  const handleApprove = useCallback(async () => {
    if (!selectedItem) return;
    setApprovingId(selectedItem.id);
    setPreviewError("");
    setActionMessage("");
    try {
      await apiFetch<{ status: "started" }>(`/jobs/review/${selectedItem.id}/approve`, {
        method: "POST",
      });
      await loadItems(page);
    } catch (e) {
      setPreviewError(getErrorDetail(e, "Failed to approve review job"));
    } finally {
      setApprovingId(null);
    }
  }, [loadItems, page, selectedItem]);

  const handleManualProcess = useCallback(async () => {
    if (!selectedItem) return;
    setManualProcessingId(selectedItem.id);
    setPreviewError("");
    setActionMessage("");
    setRemoteAlertOpen(false);
    try {
      const response = await apiFetch<{ status: "started" | "focused" }>(`/jobs/review/${selectedItem.id}/manual-process`, {
        method: "POST",
      });
      setActionMessage(
        response.status === "focused"
          ? "ManualEditor is already open and has been brought to the front."
          : "ManualEditor launch requested.",
      );
    } catch (e) {
      const detail = getErrorDetail(e, "Failed to launch ManualEditor");
      if (detail === "Remote editing is unavailable. Please use this feature on the local machine.") {
        setRemoteAlertMessage(detail);
        setRemoteAlertOpen(true);
      } else {
        setPreviewError(detail);
      }
    } finally {
      setManualProcessingId(null);
    }
  }, [selectedItem]);

  const hideMagnifier = useCallback(() => {
    setMagnifier((prev) => ({ ...prev, visible: false }));
  }, []);

  const handlePreviewMove = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const frame = previewFrameRef.current;
    const image = previewImageRef.current;
    if (!frame || !image) return;

    const frameRect = frame.getBoundingClientRect();
    const imageRect = image.getBoundingClientRect();
    const pointerX = event.clientX;
    const pointerY = event.clientY;

    if (
      pointerX < imageRect.left ||
      pointerX > imageRect.right ||
      pointerY < imageRect.top ||
      pointerY > imageRect.bottom
    ) {
      hideMagnifier();
      return;
    }

    const x = pointerX - imageRect.left;
    const y = pointerY - imageRect.top;
    const frameX = pointerX - frameRect.left;
    const frameY = pointerY - frameRect.top;
    const half = magnifierSize / 2;
    const lensX = Math.min(Math.max(frameX, half + 8), frameRect.width - half - 8);
    const lensY = Math.min(Math.max(frameY, half + 8), frameRect.height - half - 8);

    setMagnifier({
      visible: true,
      lensX,
      lensY,
      bgX: -x * magnifierZoom + half,
      bgY: -y * magnifierZoom + half,
      bgWidth: imageRect.width * magnifierZoom,
      bgHeight: imageRect.height * magnifierZoom,
    });
  }, [hideMagnifier, magnifierSize, magnifierZoom]);

  return (
    <div className="flex h-screen bg-[rgb(10,10,10)]">
      <Sidebar />

      <main className="flex-1 min-w-0 flex flex-col p-6 gap-5 overflow-hidden">
        <div className="shrink-0 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold text-[rgb(245,245,245)]">Pending Review</h1>
            <p className="mt-1 text-sm text-[rgb(163,163,163)]">
              Review O-type PLM artifacts from the VLMOCR output directory before pushing them to the remote PLM target.
            </p>
            <p className="mt-2 text-xs text-[rgb(115,115,115)] font-mono break-all">
              review_root: {reviewRoot || "\u2014"}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => void loadItems(page)}
              className="inline-flex items-center gap-2 rounded-lg border border-[rgb(64,64,64)] px-3 py-2 text-sm text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)] transition-colors"
            >
              <RefreshCcw size={16} />
              Refresh
            </button>
          </div>
        </div>

        {pageError && (
          <div className="shrink-0 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
            {pageError}
          </div>
        )}

        <section className="shrink-0 rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] p-5">
          <div className="flex items-center justify-between text-sm text-[rgb(163,163,163)]">
            <span>Total pending reviews: {total}</span>
            <span>
              Page {page} / {totalPages}
            </span>
          </div>
        </section>

        <section className="flex-1 min-h-0 grid grid-cols-[400px_minmax(0,1fr)] gap-5">
          <div className="min-h-0 flex flex-col rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] overflow-hidden">
            <div className="shrink-0 flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
              <div className="text-sm font-semibold text-[rgb(245,245,245)]">Review Queue</div>
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
              {loading ? (
                <div className="px-5 py-10 text-center text-sm text-[rgb(115,115,115)]">Loading review queue...</div>
              ) : items.length === 0 ? (
                <div className="px-5 py-10 text-center text-sm text-[rgb(115,115,115)]">No pending review jobs.</div>
              ) : (
                <div className="divide-y divide-[rgb(38,38,38)]">
                  {items.map((item) => {
                    const active = item.id === selectedId;
                    return (
                      <button
                        key={item.id}
                        onClick={() => {
                          setPreviewError("");
                          setActionMessage("");
                          setRemoteAlertOpen(false);
                          setSelectedId(item.id);
                          setMagnifier((prev) => ({ ...prev, visible: false }));
                        }}
                        className={`w-full px-5 py-4 text-left transition-colors ${
                          active ? "bg-[rgb(38,38,38)]" : "hover:bg-[rgb(38,38,38)]/40"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-sm font-medium text-[rgb(229,229,229)]">
                              {item.drawing_no || "\u2014"}
                              {item.revision ? ` / ${item.revision}` : ""}
                            </div>
                            <div className="mt-1 text-xs text-[rgb(163,163,163)] font-mono break-all">
                              {item.source_file}
                            </div>
                            <div className="mt-2 text-xs text-[rgb(115,115,115)] font-mono">
                              doc={item.docnumber || "\u2014"} ws={item.work_seq || "\u2014"}
                            </div>
                          </div>
                          <span className="inline-flex rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-300">
                            O
                          </span>
                        </div>
                        <div className="mt-3 text-xs text-[rgb(115,115,115)]">
                          Finished: {formatDateTime(item.finished_at)}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          <div className="min-h-0 flex flex-col rounded-2xl border border-[rgb(38,38,38)] bg-[rgb(23,23,23)] overflow-hidden">
            <div className="shrink-0 flex items-center justify-between px-5 py-3 border-b border-[rgb(38,38,38)]">
              <div>
                <div className="flex items-center gap-3">
                  <div className="text-sm font-semibold text-[rgb(245,245,245)]">Artifact Preview</div>
                  {previewError ? (
                    <div className="text-xs text-rose-300">{previewError}</div>
                  ) : actionMessage ? (
                    <div className="text-xs text-emerald-300">{actionMessage}</div>
                  ) : null}
                </div>
                {selectedItem ? (
                  <div className="mt-1 text-xs text-[rgb(115,115,115)] font-mono break-all">
                    {selectedItem.result_path || "\u2014"}
                  </div>
                ) : null}
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => void handleManualProcess()}
                  disabled={!selectedItem || manualProcessingId === selectedItem.id || approvingId === selectedItem.id}
                  className="inline-flex items-center gap-2 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-sm text-sky-300 hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-50 transition-colors"
                >
                  <ExternalLink size={16} />
                  {manualProcessingId === selectedItem?.id ? "Launching..." : "Manual Process"}
                </button>
                <button
                  onClick={() => void handleApprove()}
                  disabled={!selectedItem || approvingId === selectedItem.id || manualProcessingId === selectedItem?.id}
                  className="inline-flex items-center gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300 hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-50 transition-colors"
                >
                  <CheckCircle2 size={16} />
                  {approvingId === selectedItem?.id ? "Approving..." : "Approve and Push"}
                </button>
              </div>
            </div>

            {selectedItem ? (
              <div className="shrink-0 flex flex-wrap items-center justify-between gap-3 px-5 py-4 border-b border-[rgb(38,38,38)] text-xs text-[rgb(163,163,163)]">
                <div className="flex flex-wrap items-center gap-x-8 gap-y-2">
                  <div>Source: <span className="font-mono">{selectedItem.source_file}</span></div>
                  <div>Finished: <span>{formatDateTime(selectedItem.finished_at)}</span></div>
                  <div>Doc: <span className="font-mono">{selectedItem.docnumber || "\u2014"}</span></div>
                  <div>Work Seq: <span className="font-mono">{selectedItem.work_seq || "\u2014"}</span></div>
                </div>
                <div className="flex flex-wrap items-center gap-4 text-[rgb(212,212,212)]">
                  <label className="flex items-center gap-2">
                    <span className="text-[rgb(163,163,163)]">Zoom</span>
                    <input
                      type="range"
                      min="1.5"
                      max="5"
                      step="0.5"
                      value={magnifierZoom}
                      onChange={(e) => setMagnifierZoom(Number(e.target.value))}
                      className="w-28 accent-emerald-400"
                    />
                    <span className="w-10 text-right font-mono">{magnifierZoom.toFixed(1)}x</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <span className="text-[rgb(163,163,163)]">Size</span>
                    <input
                      type="range"
                      min="140"
                      max="320"
                      step="20"
                      value={magnifierSize}
                      onChange={(e) => setMagnifierSize(Number(e.target.value))}
                      className="w-28 accent-emerald-400"
                    />
                    <span className="w-12 text-right font-mono">{magnifierSize}px</span>
                  </label>
                </div>
              </div>
            ) : null}

            <div className="flex-1 min-h-0 overflow-hidden bg-[rgb(10,10,10)]">
              {!selectedItem ? (
                <div className="h-full flex items-center justify-center text-sm text-[rgb(115,115,115)]">
                  Select a pending review item.
                </div>
              ) : previewLoading ? (
                <div className="h-full flex items-center justify-center text-sm text-[rgb(115,115,115)]">
                  Loading preview...
                </div>
              ) : previewUrl ? (
                <div
                  ref={previewFrameRef}
                  onMouseMove={handlePreviewMove}
                  onMouseLeave={hideMagnifier}
                  className="relative h-full w-full p-5 flex items-center justify-center overflow-hidden"
                >
                  <img
                    ref={previewImageRef}
                    src={previewUrl}
                    alt={selectedItem.source_file}
                    className="max-w-full max-h-full w-auto h-auto rounded-xl border border-[rgb(38,38,38)] bg-white shadow-2xl cursor-zoom-in select-none"
                    draggable={false}
                  />
                  {magnifier.visible ? (
                    <div
                      className="pointer-events-none absolute border border-white/35 shadow-[0_18px_48px_rgba(0,0,0,0.45)]"
                      style={{
                        width: magnifierSize,
                        height: magnifierSize,
                        left: magnifier.lensX - magnifierSize / 2,
                        top: magnifier.lensY - magnifierSize / 2,
                        backgroundColor: "rgba(12, 12, 12, 0.92)",
                        backgroundImage: `url(${previewUrl})`,
                        backgroundRepeat: "no-repeat",
                        backgroundSize: `${magnifier.bgWidth}px ${magnifier.bgHeight}px`,
                        backgroundPosition: `${magnifier.bgX}px ${magnifier.bgY}px`,
                        borderRadius: 18,
                      }}
                    />
                  ) : null}
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-sm text-[rgb(115,115,115)]">
                  Preview unavailable.
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      <ConfirmDialog
        open={remoteAlertOpen}
        onClose={() => setRemoteAlertOpen(false)}
        onConfirm={() => setRemoteAlertOpen(false)}
        title="Manual Process Unavailable"
        description={remoteAlertMessage}
        icon={<AlertCircle size={20} className="text-amber-400 shrink-0" />}
        confirmLabel="OK"
        hideCancel
      />
    </div>
  );
}
