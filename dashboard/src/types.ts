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

export interface ConfigFieldMeta {
  type: "text" | "number" | "multi_select";
  label: string;
  category: string;
  description?: string;
  options?: string;
  step?: number;
  max_length?: number;
}

export interface ConfigCategory {
  description: string;
  icon: string;
}

export interface ConfigResponse {
  current: Record<string, unknown>;
  meta: Record<string, ConfigFieldMeta>;
  categories: Record<string, ConfigCategory>;
}
