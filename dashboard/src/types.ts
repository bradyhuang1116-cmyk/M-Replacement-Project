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
  options?: string[];
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

export type QueueSource = "api" | "watch_folder";
export type QueueStatus = "pending" | "running" | "done" | "failed";
export type PlmDeliveryStatus =
  | "not_applicable"
  | "pending"
  | "uploaded"
  | "complete"
  | "failed";

export interface QueueJobItem {
  id: number;
  source: QueueSource;
  source_file: string;
  file_path: string;
  drawing_no: string | null;
  revision: string | null;
  docnumber: string | null;
  work_seq: string | null;
  status: QueueStatus;
  retry_count: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_msg: string | null;
  result_path: string | null;
  plm_delivery_status: PlmDeliveryStatus;
  plm_remote_path: string | null;
  plm_uploaded_path: string | null;
  plm_delivery_error: string | null;
  plm_delivery_finished_at: string | null;
}

export interface QueueJobListResponse {
  items: QueueJobItem[];
  total: number;
  counts: Record<QueueStatus, number>;
  limit: number;
  offset: number;
}
