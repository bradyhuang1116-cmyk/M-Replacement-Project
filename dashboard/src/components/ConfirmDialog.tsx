"use client";

import { useEffect } from "react";

type Variant = "default" | "danger";

interface ConfirmDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  description?: string;
  icon?: React.ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: Variant;
  loading?: boolean;
  children?: React.ReactNode;
  hideCancel?: boolean;
}

const confirmBtnClass: Record<Variant, string> = {
  default:
    "bg-blue-500/15 text-blue-400 border border-blue-500/30 hover:bg-blue-500/25",
  danger:
    "bg-rose-500/15 text-rose-400 border border-rose-500/30 hover:bg-rose-500/25",
};

export default function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  icon,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "default",
  loading = false,
  children,
  hideCancel = false,
}: ConfirmDialogProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !loading) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, loading]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={loading ? undefined : onClose}
    >
      <div
        className="bg-[rgb(23,23,23)] border border-[rgb(38,38,38)] rounded-xl p-6 max-w-sm mx-4 space-y-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3">
          {icon}
          <h3 className="text-[rgb(245,245,245)] font-semibold text-base">{title}</h3>
        </div>

        {description && (
          <p className="text-sm text-[rgb(163,163,163)] leading-relaxed">{description}</p>
        )}

        {children && <div className="space-y-3">{children}</div>}

        <div className="flex gap-3 justify-end pt-1">
          {!hideCancel ? (
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="px-4 py-2 rounded-lg text-sm text-[rgb(163,163,163)] hover:bg-[rgb(38,38,38)] transition-colors disabled:opacity-40"
            >
              {cancelLabel}
            </button>
          ) : null}
          <button
            type="button"
            onClick={onConfirm}
            disabled={loading}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 ${confirmBtnClass[variant]}`}
          >
            {loading ? "Saving..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
