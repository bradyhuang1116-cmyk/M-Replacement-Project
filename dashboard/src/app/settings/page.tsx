"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import ConfirmDialog from "@/components/ConfirmDialog";
import { useConfig } from "@/hooks/useConfig";
import { Save, CheckCircle, AlertCircle, Cog, HardDrive, Database, RotateCcw, AlertTriangle, Server } from "lucide-react";

const CATEGORY_ICONS: Record<string, React.ComponentType<{ size?: number }>> = {
  Cog,
  HardDrive,
  Database,
  Server,
};

// ── inline validation helpers ──

function validateField(key: string, meta: { type: string; label: string; max_length?: number }, value: unknown): string {
  if (value === "" || value === null || value === undefined) return "";
  const keyLower = key.toLowerCase();

  // Port range validation
  if (keyLower.includes("port") && meta.type === "number") {
    const n = Number(value);
    if (!Number.isInteger(n) || n < 1 || n > 65535) return "Must be between 1 and 65535";
    return "";
  }

  // URL format validation
  if ((keyLower.includes("url") || keyLower.includes("base_url")) && meta.type === "text") {
    const s = String(value).trim();
    if (s && !/^https?:\/\/.+/.test(s)) return "Must start with http:// or https://";
    return "";
  }

  // Number bounds check
  if (meta.type === "number") {
    const n = Number(value);
    if (!isFinite(n)) return "Must be a valid number";
    return "";
  }

  // Text max_length check
  if (meta.type === "text" && meta.max_length) {
    const s = String(value);
    if (s.length > meta.max_length) return `Max ${meta.max_length} characters`;
    return "";
  }

  return "";
}

function hasValidationErrors(errors: Record<string, string>): boolean {
  return Object.values(errors).some((v) => v !== "");
}

export default function SettingsPage() {
  const { config, loading, saving, error, successMsg, saveConfig, reload } = useConfig();
  const [dirtyValues, setDirtyValues] = useState<Record<string, unknown>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [showConfirm, setShowConfirm] = useState(false);
  const [restarting, setRestarting] = useState(false);

  useEffect(() => {
    if (config) {
      setDirtyValues({});
      setHasChanges(false);
      setValidationErrors({});
    }
  }, [config]);

  function handleFieldChange(key: string, value: unknown) {
    const next = { ...dirtyValues, [key]: value };
    setDirtyValues(next);
    setHasChanges(true);

    // Validate the field that changed
    if (config?.meta[key]) {
      const err = validateField(key, config.meta[key], value);
      setValidationErrors((prev) => ({ ...prev, [key]: err }));
    }
  }

  function getEffectiveValue(key: string): unknown {
    if (key in dirtyValues) return dirtyValues[key];
    return config?.current[key] ?? "";
  }

  // Re-validate all dirty fields whenever config or dirty values change
  const canSave = useMemo(() => {
    if (!hasChanges) return false;
    return !hasValidationErrors(validationErrors);
  }, [hasChanges, validationErrors]);

  const handleSaveClick = () => {
    // Re-validate all dirty fields
    const errors: Record<string, string> = {};
    for (const key of Object.keys(dirtyValues)) {
      if (config?.meta[key]) {
        const err = validateField(key, config.meta[key], dirtyValues[key]);
        if (err) errors[key] = err;
      }
    }
    setValidationErrors(errors);
    if (hasValidationErrors(errors)) return;
    setShowConfirm(true);
  };

  const handleConfirmSave = async () => {
    setShowConfirm(false);
    if (!hasChanges || Object.keys(dirtyValues).length === 0) return;
    try {
      await saveConfig(dirtyValues);
      setDirtyValues({});
      setHasChanges(false);
      setValidationErrors({});
      reload();
    } catch {
      // error handled by hook
    }
  };

  const handleRestart = useCallback(async () => {
    setRestarting(true);
    try {
      await fetch("http://localhost:8000/api/v1/system/restart", { method: "POST" });
    } catch {
      // Expected: backend exits before response completes
    }
    setTimeout(() => {
      setRestarting(false);
      window.location.reload();
    }, 5000);
  }, []);

  const categories = useMemo(() => {
    if (!config) return [];
    return Object.entries(config.categories).map(([catName, catMeta]) => {
      const fields = Object.entries(config.meta).filter(
        ([, fieldMeta]) => fieldMeta.category === catName
      );
      return { name: catName, ...catMeta, fields };
    });
  }, [config]);

  return (
    <div className="flex h-screen bg-[rgb(10,10,10)]">
      <Sidebar />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Scrollable content area */}
        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="p-6 space-y-6 pb-16">
            <h1 className="text-lg font-semibold text-[rgb(245,245,245)] shrink-0">Settings</h1>

            {/* Success banner with optional restart */}
            {/* {successMsg && (
              <div className="shrink-0 flex items-center justify-between px-4 py-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle size={16} />
                  {successMsg}
                </div>
                <button
                  onClick={handleRestart}
                  disabled={restarting}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 border border-emerald-500/30 transition-colors disabled:opacity-50"
                >
                  <RotateCcw size={14} className={restarting ? "animate-spin" : ""} />
                  {restarting ? "Restarting..." : "Restart Now"}
                </button>
              </div>
            )} */}

            {error && (
              <div className="shrink-0 flex items-center gap-2 px-4 py-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            {loading && !config ? (
              <div className="text-center py-12 text-sm text-[rgb(115,115,115)]">Loading...</div>
            ) : !config ? null : (
              <div className="grid grid-cols-1 gap-5">
                {categories.length === 0 ? (
                  <div className="text-center py-12 text-sm text-[rgb(115,115,115)]">
                    No configuration fields found.
                  </div>
                ) : (
                  categories.map((cat) => {
                    const IconComp = CATEGORY_ICONS[cat.icon] || Cog;
                    return (
                      <div
                        key={cat.name}
                        className="bg-[rgb(23,23,23)] rounded-xl border border-[rgb(38,38,38)] p-5"
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-[rgb(163,163,163)]"><IconComp size={16} /></span>
                          <h2 className="text-sm font-semibold text-[rgb(245,245,245)]">{cat.name}</h2>
                          <p className="text-xs text-[rgb(115,115,115)] ml-1">{cat.description}</p>
                        </div>

                        <div className="space-y-5">
                          {cat.fields.length === 0 ? (
                            <p className="text-xs text-[rgb(115,115,115)]">No fields in this category.</p>
                          ) : (
                            cat.fields.map(([key, meta]) => {
                              const err = validationErrors[key];
                              return (
                                <div key={key}>
                                  <label className="text-sm font-medium text-[rgb(212,212,212)] mb-1 block">
                                    {meta.label}
                                  </label>
                                  {meta.description && (
                                    <p className="text-xs text-[rgb(115,115,115)] mb-2">{meta.description}</p>
                                  )}

                                  {meta.type === "number" ? (
                                    <input
                                      type="number"
                                      value={Number(getEffectiveValue(key))}
                                      onChange={(e) => handleFieldChange(key, e.target.valueAsNumber)}
                                      step={meta.step ?? 1}
                                      className={`w-full max-w-xs px-3 py-2 text-sm bg-[rgb(38,38,38)] border rounded-lg text-[rgb(229,229,229)] focus:outline-none transition-colors ${
                                        err
                                          ? "border-rose-500/60 focus:border-rose-500"
                                          : "border-[rgb(64,64,64)] focus:border-[rgb(82,82,82)]"
                                      }`}
                                    />
                                  ) : (
                                    <input
                                      type="text"
                                      value={String(getEffectiveValue(key) ?? "")}
                                      onChange={(e) => handleFieldChange(key, e.target.value)}
                                      maxLength={meta.max_length}
                                      className={`w-full max-w-lg px-3 py-2 text-sm bg-[rgb(38,38,38)] border rounded-lg text-[rgb(229,229,229)] focus:outline-none transition-colors ${
                                        err
                                          ? "border-rose-500/60 focus:border-rose-500"
                                          : "border-[rgb(64,64,64)] focus:border-[rgb(82,82,82)]"
                                      }`}
                                    />
                                  )}

                                  {err && (
                                    <p className="mt-1 text-xs text-rose-400 flex items-center gap-1">
                                      <AlertCircle size={12} />
                                      {err}
                                    </p>
                                  )}
                                </div>
                              );
                            })
                          )}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            )}
          </div>
        </div>

        {/* Sticky bottom bar */}
        {config && (
          <div className="shrink-0 sticky bottom-0 bg-[rgb(10,10,10)] border-t border-[rgb(38,38,38)] flex items-center justify-end gap-3 px-6 h-16">
            {hasChanges && (
              <span className="text-xs text-[rgb(115,115,115)]">
                Changes will be applied immediately
              </span>
            )}
            <button
              onClick={handleSaveClick}
              disabled={!canSave || saving}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-lg"
            >
              <Save size={18} />
              {saving ? "Saving..." : "Save Changes"}
            </button>
          </div>
        )}
      </main>

      <ConfirmDialog
        open={showConfirm}
        onClose={() => setShowConfirm(false)}
        onConfirm={handleConfirmSave}
        title="Save Configuration"
        description="Saved settings need a backend restart to apply."
        icon={<AlertTriangle size={20} className="text-amber-400 shrink-0" />}
        confirmLabel="Save"
        loading={saving}
      />
    </div>
  );
}
