"use client";

import { useState, useEffect, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import { useConfig } from "@/hooks/useConfig";
import { Save, CheckCircle, AlertCircle, Cog, HardDrive, Database } from "lucide-react";

const CATEGORY_ICONS: Record<string, React.ComponentType<{ size?: number }>> = {
  Cog,
  HardDrive,
  Database,
};

export default function SettingsPage() {
  const { config, loading, saving, error, successMsg, saveConfig, reload } = useConfig();
  const [dirtyValues, setDirtyValues] = useState<Record<string, unknown>>({});
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (config) {
      setDirtyValues({});
      setHasChanges(false);
    }
  }, [config]);

  function handleFieldChange(key: string, value: unknown) {
    setDirtyValues((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
  }

  function getEffectiveValue(key: string): unknown {
    if (key in dirtyValues) return dirtyValues[key];
    return config?.current[key] ?? "";
  }

  async function handleSave() {
    if (!hasChanges || Object.keys(dirtyValues).length === 0) return;
    try {
      await saveConfig(dirtyValues);
      setDirtyValues({});
      setHasChanges(false);
      reload();
    } catch {
      // error handled by hook
    }
  }

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

      <main className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6 pb-16">
          <h1 className="text-lg font-semibold text-[rgb(245,245,245)]">Settings</h1>

          {successMsg && (
            <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm">
              <CheckCircle size={16} />
              {successMsg}
            </div>
          )}
          {error && (
            <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          {loading && !config ? (
            <div className="text-center py-12 text-sm text-[rgb(115,115,115)]">Loading...</div>
          ) : !config ? null : (
            <div className="grid grid-cols-1 gap-5">
              {categories.map((cat) => {
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
                      {cat.fields.map(([key, meta]) => (
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
                              className="w-full max-w-xs px-3 py-2 text-sm bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-[rgb(229,229,229)] focus:outline-none focus:border-[rgb(82,82,82)]"
                            />
                          ) : (
                            <input
                              type="text"
                              value={String(getEffectiveValue(key) ?? "")}
                              onChange={(e) => handleFieldChange(key, e.target.value)}
                              maxLength={meta.max_length}
                              className="w-full max-w-lg px-3 py-2 text-sm bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-[rgb(229,229,229)] focus:outline-none focus:border-[rgb(82,82,82)]"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Sticky bottom bar — only spans main content, not sidebar */}
        {config && (
          <div className="sticky bottom-0 bg-[rgb(10,10,10)] flex items-center justify-end gap-3 px-6 h-25">
            {hasChanges && (
              <span className="text-xs text-[rgb(115,115,115)]">
                Unsaved changes — will take effect after backend restart
              </span>
            )}
            <button
              onClick={handleSave}
              disabled={!hasChanges || saving}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-lg"
            >
              <Save size={18} />
              {saving ? "Saving..." : "Save Changes"}
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
