"use client";

import { useCallback, useState, useRef, RefObject } from "react";
import { Upload, Video, RefreshCw } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface VideoUploaderProps {
  onUpload: (file: File) => void;
  isProcessing: boolean;
  previewUrl?: string;
  videoRef?: RefObject<HTMLVideoElement | null>;
  /** File metadata passed from parent when upload originates outside this component */
  externalFileName?: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function VideoUploader({ onUpload, isProcessing, previewUrl, videoRef, externalFileName }: VideoUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileSize, setFileSize] = useState<string | null>(null);
  const [fileType, setFileType] = useState<string | null>(null);
  const [hovered, setHovered] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const replaceRef = useRef<HTMLInputElement>(null);

  const displayName = fileName || externalFileName || null;

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("video/")) return;
      setFileName(file.name);
      setFileSize(formatBytes(file.size));
      setFileType(file.type.split("/")[1]?.toUpperCase() || "VIDEO");
      onUpload(file);
    },
    [onUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  /** Reset an input so selecting the same file again triggers onChange */
  const resetInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  return (
    <div
      className="card-surface flex flex-col h-full overflow-hidden"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-2 px-5 pt-4 pb-2">
        {previewUrl && (
          <span className="relative flex h-1.5 w-1.5">
            <span className="live-dot inline-flex h-full w-full rounded-full bg-[#C75B6E]" />
          </span>
        )}
        <span className="section-label">Live Input</span>
        {previewUrl && (
          <span className="ml-auto text-[11px] text-[rgba(245,247,250,0.44)] font-medium">
            {isProcessing ? "Processing" : "Active"}
          </span>
        )}
      </div>

      <div className="flex-1 px-4 pb-4 flex flex-col">
        <AnimatePresence mode="wait">
          {previewUrl ? (
            <motion.div
              key="preview"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              className="flex flex-col flex-1"
            >
              {/* Video */}
              <div className="media-frame relative">
                <motion.div
                  animate={{ scale: hovered ? 1.02 : 1 }}
                  transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                >
                  <video
                    ref={videoRef}
                    src={previewUrl}
                    controls
                    autoPlay
                    loop
                    muted
                    className="w-full object-cover bg-[#15181D]"
                    style={{ aspectRatio: "16/9" }}
                  />
                </motion.div>
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#111315]/80 to-transparent rounded-b-[1rem] px-3 py-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-[rgba(245,247,250,0.48)] truncate font-medium">
                      {displayName || "Input video"}
                    </span>
                    <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-md ${
                      isProcessing
                        ? "text-[#F4B267] bg-[rgba(244,178,103,0.1)]"
                        : "text-[#6FD0A3] bg-[rgba(111,208,163,0.1)]"
                    }`}>
                      {isProcessing ? "Processing" : "Ready"}
                    </span>
                  </div>
                </div>
              </div>

              {/* File metadata + actions */}
              <div className="mt-3 flex flex-col gap-3 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  {fileType && (
                    <span className="px-2 py-0.5 rounded-md text-[10px] font-semibold tracking-wide bg-[rgba(199,91,110,0.1)] text-[#D98090] border border-[rgba(199,91,110,0.15)]">
                      {fileType}
                    </span>
                  )}
                  {fileSize && (
                    <span className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-[rgba(255,255,255,0.04)] text-[rgba(245,247,250,0.52)] border border-[rgba(255,255,255,0.06)]">
                      {fileSize}
                    </span>
                  )}
                  <span className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-[rgba(255,255,255,0.04)] text-[rgba(245,247,250,0.52)] border border-[rgba(255,255,255,0.06)]">
                    Looped
                  </span>
                </div>

                <div className="flex flex-col gap-2 mt-auto">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-[rgba(245,247,250,0.44)] font-medium">Source</span>
                    <span className="text-[11px] text-[rgba(245,247,250,0.68)] font-medium truncate ml-3 max-w-[140px]">
                      {displayName || "Demo clip"}
                    </span>
                  </div>
                  <div className="h-px bg-[rgba(255,255,255,0.06)]" />

                  <label className="flex items-center justify-center gap-1.5 py-2 rounded-lg text-[11px] font-medium text-[rgba(245,247,250,0.52)] bg-[rgba(255,255,255,0.04)] border border-[rgba(255,255,255,0.06)] hover:bg-[rgba(255,255,255,0.07)] hover:text-[rgba(245,247,250,0.72)] transition-all cursor-pointer">
                    <RefreshCw className="h-3 w-3" />
                    Replace Clip
                    <input
                      ref={replaceRef}
                      type="file"
                      accept="video/*"
                      className="hidden"
                      onChange={resetInput}
                    />
                  </label>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              className={`flex flex-col items-center justify-center rounded-xl cursor-pointer transition-all duration-300 flex-1 min-h-[200px] border border-dashed ${
                dragOver
                  ? "bg-[rgba(199,91,110,0.08)] border-[rgba(199,91,110,0.3)]"
                  : "bg-[#15181D] border-[rgba(255,255,255,0.08)] hover:bg-[#171A20] hover:border-[rgba(255,255,255,0.14)]"
              }`}
            >
              {dragOver ? (
                <Upload className="h-6 w-6 text-[#C75B6E] mb-3" />
              ) : (
                <Video className="h-6 w-6 text-[rgba(245,247,250,0.24)] mb-3" />
              )}
              <p className="text-sm text-[rgba(245,247,250,0.6)]">Drop a clip or browse</p>
              <p className="text-[11px] text-[rgba(245,247,250,0.28)] mt-1">MP4, AVI, MOV</p>
              <input
                ref={inputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={resetInput}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
