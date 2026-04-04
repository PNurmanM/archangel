export type AlertLevel = "NORMAL" | "WATCH" | "ALERT";

export interface BrainRegion {
  rank: number;
  name: string;
  label: string;
  score: number;
}

export interface SystemScore {
  system: string;
  score: number;
}

export interface TimePoint {
  time: number;
  total: number;
  systems: Record<string, number>;
}

export interface BrainPrediction {
  /** Input video path */
  inputVideoUrl: string;
  /** Brain movie MP4 path */
  brainMovieUrl: string;
  /** Mean brain PNG */
  meanBrainUrl: string;
  /** Peak brain PNG */
  peakBrainUrl: string;
  /** System scores */
  systemScores: SystemScore[];
  alertLevel: AlertLevel;
  alertLabel: string;
  topRegions: BrainRegion[];
  summary: string;
  totalActivity: number;
  dominantSystem: string;
  engagement: string;
  timeline: TimePoint[];
}

export interface UploadResponse {
  taskId: string;
  status: "processing" | "complete" | "error";
  result?: BrainPrediction;
}
