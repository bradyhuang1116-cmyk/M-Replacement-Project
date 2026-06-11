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

export type ProcessLogStatus = "success" | "failed";
export type ProcessMode = "O" | "N";

export interface ProcessLogItem {
  id: number;
  drawing_no: string;
  revision: string | null;
  filename: string;
  ocr_flag: ProcessMode;
  status: ProcessLogStatus;
  process_date: string;
  docnumber: string | null;
  work_seq: string | null;
}

export interface PaginatedLogs {
  items: ProcessLogItem[];
  total: number;
}
