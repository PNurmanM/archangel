import { UploadResponse } from "./types";
import { generatePredictionFromAnalysis } from "./mock-data";
import { analyzeVideoFrames } from "./video-analyzer";
import { generateBrainMovie } from "./brain-movie-generator";

/** Change this to your backend URL when ready */
const API_BASE = "http://localhost:8000";

const USE_MOCK = true;

export async function uploadVideo(
  file: File,
  onProgress?: (stage: string, pct: number) => void,
): Promise<UploadResponse> {
  if (USE_MOCK) {
    onProgress?.("Analyzing frames", 0);

    // 1. Analyse actual video frames
    const { duration, frames } = await analyzeVideoFrames(file);
    onProgress?.("Analyzing frames", 1);

    // 2. Generate brain-activation movie (real-time recording)
    onProgress?.("Generating brain movie", 0);
    const brainMovieUrl = await generateBrainMovie(frames, duration, (pct) => {
      onProgress?.("Generating brain movie", pct);
    });

    // 3. Build prediction from analysis
    const videoUrl = URL.createObjectURL(file);
    const result = generatePredictionFromAnalysis(frames, duration, videoUrl);
    result.brainMovieUrl = brainMovieUrl;

    return { taskId: `mock-${Date.now()}`, status: "complete", result };
  }

  const formData = new FormData();
  formData.append("video", file);
  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}
