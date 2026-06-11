"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface DatePickerProps {
  value: string;
  onChange: (value: string) => void;
  label: string;
}

const WEEKDAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];

function getMonthDays(year: number, month: number): { day: number; empty: boolean }[] {
  const first = new Date(year, month, 1).getDay();
  const total = new Date(year, month + 1, 0).getDate();
  const cells: { day: number; empty: boolean }[] = [];
  for (let i = 0; i < first; i++) cells.push({ day: 0, empty: true });
  for (let d = 1; d <= total; d++) cells.push({ day: d, empty: false });
  return cells;
}

function formatDisplayDate(dateStr: string): string {
  if (!dateStr) return "";
  const [y, m, d] = dateStr.split("-");
  return `${y}/${m}/${d}`;
}

export default function DatePicker({ value, onChange, label }: DatePickerProps) {
  const [open, setOpen] = useState(false);
  const [year, setYear] = useState(0);
  const [month, setMonth] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialise navigation to the selected date or today
  useEffect(() => {
    if (value) {
      const d = new Date(value + "T00:00:00");
      setYear(d.getFullYear());
      setMonth(d.getMonth());
    } else {
      const d = new Date();
      setYear(d.getFullYear());
      setMonth(d.getMonth());
    }
  }, [value]);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handle = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [open]);

  const handleSelect = useCallback(
    (day: number) => {
      const m = String(month + 1).padStart(2, "0");
      const d = String(day).padStart(2, "0");
      onChange(`${year}-${m}-${d}`);
      setOpen(false);
    },
    [year, month, onChange]
  );

  const prevMonth = () => {
    if (month === 0) { setYear((y) => y - 1); setMonth(11); }
    else setMonth((m) => m - 1);
  };

  const nextMonth = () => {
    if (month === 11) { setYear((y) => y + 1); setMonth(0); }
    else setMonth((m) => m + 1);
  };

  const isToday = (day: number) => {
    if (!value && !open) return false;
    const d = new Date();
    return d.getFullYear() === year && d.getMonth() === month && d.getDate() === day;
  };

  const isSelected = (day: number) => {
    if (!value) return false;
    const [sy, sm, sd] = value.split("-").map(Number);
    return sy === year && sm === month + 1 && sd === day;
  };

  const cells = getMonthDays(year, month);
  const MONTH_NAMES = "January February March April May June July August September October November December".split(" ");

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-xs font-medium text-[rgb(163,163,163)] mb-1.5">
        {label}
      </label>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-sm bg-[rgb(38,38,38)] border border-[rgb(64,64,64)] rounded-lg text-left focus:outline-none focus:border-[rgb(82,82,82)] transition-colors cursor-pointer"
      >
        <span className={value ? "text-[rgb(229,229,229)]" : "text-[rgb(115,115,115)]"}>
          {value ? formatDisplayDate(value) : "Select date..."}
        </span>
        <svg
          className={`w-4 h-4 text-[rgb(115,115,115)] transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute z-50 mt-1 w-[280px] rounded-xl border border-[rgb(64,64,64)] bg-[rgb(23,23,23)] p-4 shadow-2xl">
          {/* Month/Year header */}
          <div className="flex items-center justify-between mb-3">
            <button
              type="button"
              onClick={prevMonth}
              className="p-1 rounded-lg hover:bg-[rgb(38,38,38)] text-[rgb(163,163,163)] hover:text-[rgb(229,229,229)] transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-sm font-medium text-[rgb(229,229,229)]">
              {MONTH_NAMES[month]} {year}
            </span>
            <button
              type="button"
              onClick={nextMonth}
              className="p-1 rounded-lg hover:bg-[rgb(38,38,38)] text-[rgb(163,163,163)] hover:text-[rgb(229,229,229)] transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>

          {/* Weekday headers */}
          <div className="grid grid-cols-7 mb-1">
            {WEEKDAYS.map((w) => (
              <div key={w} className="text-center text-xs font-medium text-[rgb(115,115,115)] py-1">
                {w}
              </div>
            ))}
          </div>

          {/* Day grid */}
          <div className="grid grid-cols-7 gap-0.5">
            {cells.map((cell, i) =>
              cell.empty ? (
                <div key={`e-${i}`} />
              ) : (
                <button
                  key={`d-${cell.day}`}
                  type="button"
                  onClick={() => handleSelect(cell.day)}
                  className={`w-full aspect-square flex items-center justify-center text-xs rounded-lg transition-colors ${
                    isSelected(cell.day)
                      ? "bg-blue-600 text-white font-semibold"
                      : isToday(cell.day)
                      ? "bg-blue-500/20 text-blue-400 font-medium"
                      : "text-[rgb(212,212,212)] hover:bg-[rgb(38,38,38)]"
                  }`}
                >
                  {cell.day}
                </button>
              )
            )}
          </div>

          {/* Bottom: Clear / Today */}
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-[rgb(38,38,38)]">
            <button
              type="button"
              onClick={() => { onChange(""); setOpen(false); }}
              className="text-xs text-[rgb(115,115,115)] hover:text-[rgb(229,229,229)] transition-colors"
            >
              Clear
            </button>
            <button
              type="button"
              onClick={() => {
                const d = new Date();
                const y = d.getFullYear();
                const m = String(d.getMonth() + 1).padStart(2, "0");
                const day = String(d.getDate()).padStart(2, "0");
                onChange(`${y}-${m}-${day}`);
                setOpen(false);
              }}
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              Today
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
