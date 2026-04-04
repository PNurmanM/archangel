import { BrainPrediction, TimePoint, SystemScore, BrainRegion, AlertLevel } from "./types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || `ws://${typeof window !== "undefined" ? window.location.hostname : "localhost"}:8000/ws`;
const FRAME_URL = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/frame`
  : `http://${typeof window !== "undefined" ? window.location.hostname : "localhost"}:8000/api/frame`;

export interface LivePrediction {
  status: string;
  inference_count: number;
  history_length: number;
  systemScores: SystemScore[];
  alertLevel: AlertLevel;
  alertLabel: string;
  topRegions: BrainRegion[];
  totalActivity: number;
  dominantSystem: string;
  engagement: string;
  timeline: TimePoint[];
  spike: string;
  spike_pct: number;
  emotions: Record<string, number>;
  timing_ms: number;
}

type OnPrediction = (pred: LivePrediction) => void;
type OnStatus = (status: "connecting" | "connected" | "disconnected" | "buffering") => void;

export function connectLiveStream(
  onPrediction: OnPrediction,
  onStatus: OnStatus,
): () => void {
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;

  function connect() {
    if (stopped) return;
    onStatus("connecting");

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      onStatus("connected");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("[WS] received:", data.status, data.inference_count);
        if (data.status === "buffering") {
          onStatus("buffering");
          return;
        }
        if (data.status === "ok") {
          onPrediction(data as LivePrediction);
        }
      } catch (e) {
        console.error("[WS] parse error:", e);
      }
    };

    ws.onclose = (e) => {
      console.log("[WS] closed:", e.code, e.reason);
      if (!stopped) {
        onStatus("disconnected");
        reconnectTimer = setTimeout(connect, 2000);
      }
    };

    ws.onerror = (e) => {
      console.error("[WS] error:", e);
      ws?.close();
    };
  }

  connect();

  // Return cleanup function
  return () => {
    stopped = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    ws?.close();
  };
}

export function getFrameUrl(): string {
  return `${FRAME_URL}?t=${Date.now()}`;
}
