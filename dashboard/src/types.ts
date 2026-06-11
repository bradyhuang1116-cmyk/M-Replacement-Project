export type FileStatus = "completed" | "processing" | "pending" | "failed";
export type ProcessingMethod = "VLMOCR" | "PDF Replacement" | "";

export interface FileItem {
  id: string;
  filename: string;
  status: FileStatus;
  duration: string;
  method: ProcessingMethod;
  replacedFilename: string;
  error: string;
}

export interface GpuMetrics {
  name: string;
  index: number;
  temperature: number;
  gpuLoad: number;
  memoryUsed: number;
  memoryTotal: number;
  fanSpeed: number;
  clockSpeed: number;
  powerDraw: number;
  powerLimit: number;
}

export interface JobState {
  isRunning: boolean;
  phase: string;
  currentFile: number;
  totalFiles: number;
  elapsedSeconds: number;
  progress: number;
  logs: string[];
}
