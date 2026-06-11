"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { JobState, FileItem } from "@/types";
import { apiSSE } from "@/lib/api";

interface SSEPayload {
  phase: string;
  isRunning: boolean;
  currentFile: number;
  totalFiles: number;
  elapsedSeconds: number;
  totalElapsed: number;
  progress: number;
  logs: string[];
  files: {
    filename: string;
    status: string;
    duration: string;
    method: string;
    replacedFilename: string;
    error: string;
  }[];
}

const defaultJob: JobState = {
  isRunning: false,
  phase: "idle",
  currentFile: 0,
  totalFiles: 0,
  elapsedSeconds: 0,
  progress: 0,
  logs: [],
};

export function useJobStatus() {
  const [job, setJob] = useState<JobState>(defaultJob);
  const [files, setFiles] = useState<FileItem[]>([]);
  const esRef = useRef<EventSource | null>(null);

  const connect = useCallback(() => {
    if (esRef.current) esRef.current.close();

    const es = apiSSE("/jobs/status");
    esRef.current = es;

    es.onmessage = (event) => {
      try {
        const data: SSEPayload = JSON.parse(event.data);
        setJob({
          isRunning: data.isRunning,
          phase: data.phase,
          currentFile: data.currentFile,
          totalFiles: data.totalFiles,
          elapsedSeconds: data.elapsedSeconds,
          progress: data.progress,
          logs: data.logs || [],
        });
        setFiles(
          data.files.map((f, i) => ({
            id: String(i + 1),
            filename: f.filename,
            status: f.status as FileItem["status"],
            duration: f.duration,
            method: (f.method || "") as FileItem["method"],
            replacedFilename: f.replacedFilename,
            error: f.error || "",
          }))
        );
      } catch {}
    };

    es.onerror = () => {
      es.close();
      setTimeout(connect, 3000);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (esRef.current) esRef.current.close();
    };
  }, [connect]);

  return { job, files };
}
