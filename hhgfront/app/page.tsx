"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, useMotionValue, useTransform, useSpring, useScroll, useMotionValueEvent } from "framer-motion";
import { Play, Upload, Cpu, Eye, FileText, ArrowRight } from "lucide-react";
import { ArchAngelLogo } from "@/components/logo";
import { VideoUploader } from "@/components/video-uploader";
import { BrainViewer } from "@/components/brain-viewer";
import { SnapshotRail } from "@/components/snapshot-rail";
import { ActivityChart } from "@/components/activity-chart";
import { SystemCapsules } from "@/components/system-capsules";
import { RegionList } from "@/components/region-table";
import { SummaryPanel } from "@/components/summary-panel";
import { uploadVideo } from "@/lib/api";
import { mockPrediction } from "@/lib/mock-data";
import { BrainPrediction } from "@/lib/types";

function Skeleton({ className }: { className?: string }) {
  return <div className={`skeleton ${className ?? ""}`} />;
}

const ease = [0.16, 1, 0.3, 1] as const;

/* ── Neural background SVG for hero ── */
function NeuralBackground() {
  return (
    <svg
      className="absolute inset-0 w-full h-full opacity-[0.07]"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id="fadeEdge" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="white" stopOpacity="1" />
          <stop offset="70%" stopColor="white" stopOpacity="0.3" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </radialGradient>
        <mask id="fadeMask">
          <rect width="100%" height="100%" fill="url(#fadeEdge)" />
        </mask>
      </defs>
      <g mask="url(#fadeMask)" stroke="#C75B6E" strokeWidth="0.5" fill="none">
        <path d="M200,100 Q350,180 500,150 T800,200" opacity="0.5" />
        <path d="M100,250 Q300,200 450,280 T750,250" opacity="0.4" />
        <path d="M300,50 Q400,120 550,80 T900,130" opacity="0.3" />
        <path d="M150,350 Q350,300 500,370 T850,320" opacity="0.35" />
        <path d="M50,180 Q250,150 400,210 T700,170" opacity="0.3" />
        <line x1="350" y1="180" x2="450" y2="280" opacity="0.2" />
        <line x1="500" y1="150" x2="550" y2="80" opacity="0.2" />
        <line x1="400" y1="210" x2="500" y2="150" opacity="0.15" />
        <line x1="300" y1="200" x2="350" y2="300" opacity="0.15" />
        <circle cx="350" cy="180" r="3" fill="#C75B6E" opacity="0.4" />
        <circle cx="500" cy="150" r="4" fill="#A04458" opacity="0.35" />
        <circle cx="450" cy="280" r="3" fill="#D98090" opacity="0.3" />
        <circle cx="550" cy="80" r="2.5" fill="#C75B6E" opacity="0.3" />
        <circle cx="400" cy="210" r="3" fill="#A04458" opacity="0.25" />
        <circle cx="700" cy="170" r="2" fill="#D98090" opacity="0.25" />
        <circle cx="250" cy="200" r="2.5" fill="#C75B6E" opacity="0.2" />
        <circle cx="800" cy="200" r="2" fill="#A04458" opacity="0.2" />
      </g>
    </svg>
  );
}

export default function Page() {
  const [prediction, setPrediction] = useState<BrainPrediction | null>(mockPrediction);
  const [isProcessing, setIsProcessing] = useState(false);
  /* User-uploaded video URL — kept separate from prediction so it persists after mock API returns */
  const [userVideoUrl, setUserVideoUrl] = useState<string | null>(null);
  const [userFileName, setUserFileName] = useState<string | null>(null);
  const [processingStage, setProcessingStage] = useState<string>("");

  const inputVideoRef = useRef<HTMLVideoElement>(null);
  const brainVideoRef = useRef<HTMLVideoElement>(null);
  const heroRef = useRef<HTMLDivElement>(null);

  /* Sticky nav state */
  const { scrollY } = useScroll();
  const [showStickyNav, setShowStickyNav] = useState(false);
  useMotionValueEvent(scrollY, "change", (y) => {
    setShowStickyNav(y > 400);
  });

  const mouseX = useMotionValue(0.5);
  const mouseY = useMotionValue(0.5);
  const springX = useSpring(mouseX, { stiffness: 50, damping: 20 });
  const springY = useSpring(mouseY, { stiffness: 50, damping: 20 });

  const heroGlowX = useTransform(springX, [0, 1], ["-10%", "10%"]);
  const heroGlowY = useTransform(springY, [0, 1], ["-8%", "8%"]);
  const textX = useTransform(springX, [0, 1], [4, -4]);
  const textY = useTransform(springY, [0, 1], [3, -3]);
  const neuralX = useTransform(springX, [0, 1], [8, -8]);
  const neuralY = useTransform(springY, [0, 1], [6, -6]);

  const handleHeroMouse = useCallback(
    (e: React.MouseEvent) => {
      if (!heroRef.current) return;
      const rect = heroRef.current.getBoundingClientRect();
      mouseX.set((e.clientX - rect.left) / rect.width);
      mouseY.set((e.clientY - rect.top) / rect.height);
    },
    [mouseX, mouseY]
  );

  /* Card mouse highlight tracking */
  useEffect(() => {
    function trackMouse(e: MouseEvent) {
      const cards = document.querySelectorAll<HTMLElement>(".card, .card-elevated, .card-surface");
      cards.forEach((card) => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--mouse-x", `${e.clientX - rect.left}px`);
        card.style.setProperty("--mouse-y", `${e.clientY - rect.top}px`);
      });
    }
    window.addEventListener("mousemove", trackMouse);
    return () => window.removeEventListener("mousemove", trackMouse);
  }, []);

  /* Sync input video ↔ brain video playback.
     When the input is longer than the brain movie, wrap with modulo so the
     brain movie loops smoothly instead of glitching past its end. */
  useEffect(() => {
    if (!prediction) return;
    let rafId: number;
    const sync = () => {
      const input = inputVideoRef.current;
      const brain = brainVideoRef.current;
      if (input && brain && brain.duration && Number.isFinite(brain.duration)) {
        const targetTime = brain.duration > 0
          ? input.currentTime % brain.duration
          : input.currentTime;
        if (Math.abs(targetTime - brain.currentTime) > 0.15) {
          brain.currentTime = targetTime;
        }
        if (!input.paused && brain.paused) brain.play();
        if (input.paused && !brain.paused) brain.pause();
      }
      rafId = requestAnimationFrame(sync);
    };
    rafId = requestAnimationFrame(sync);
    return () => cancelAnimationFrame(rafId);
  }, [prediction]);

  const handleUpload = useCallback(async (file: File) => {
    const url = URL.createObjectURL(file);
    setUserVideoUrl(url);
    setUserFileName(file.name);
    setIsProcessing(true);
    setProcessingStage("Analyzing frames");
    setPrediction(null);
    try {
      const res = await uploadVideo(file, (stage, _pct) => {
        setProcessingStage(stage);
      });
      if (res.result) setPrediction(res.result);
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setIsProcessing(false);
      setProcessingStage("");
    }
  }, []);

  const handleDemo = useCallback(() => {
    /* Clear user video so demo assets show */
    setUserVideoUrl(null);
    setUserFileName(null);
    setPrediction(null);
    setIsProcessing(true);
    setTimeout(() => {
      setPrediction(mockPrediction);
      setIsProcessing(false);
    }, 2000);
  }, []);

  /** Helper for all file inputs — handles file + resets value so same file can be re-selected */
  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleUpload(file);
      e.target.value = "";
    },
    [handleUpload]
  );

  const systemNames = prediction ? prediction.systemScores.map((s) => s.system) : [];
  /* User video takes priority over prediction demo video */
  const displayVideoUrl = userVideoUrl || prediction?.inputVideoUrl;

  return (
    <div className="min-h-screen bg-[#111315]">
      {/* ═══════════ STICKY NAV (appears on scroll) ═══════════ */}
      <motion.nav
        initial={false}
        animate={{
          y: showStickyNav ? 0 : -80,
          opacity: showStickyNav ? 1 : 0,
        }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-[#111315]/80 border-b border-[rgba(255,255,255,0.06)]"
      >
        <div className="mx-auto max-w-[1320px] px-5 md:px-8 xl:px-10 flex items-center justify-between h-14">
          <div className="flex items-center gap-2.5">
            <ArchAngelLogo size={24} />
            <span className="text-[14px] font-semibold text-[#F5F7FA] tracking-tight">
              ArchAngel
            </span>
          </div>
          <div className="flex items-center gap-2.5">
            <label className="btn-secondary flex items-center gap-1.5 px-3.5 py-1.5 text-[12px] cursor-pointer">
              <Upload className="h-3 w-3" />
              Upload Clip
              <input type="file" accept="video/*" className="hidden" onChange={handleFileInput} />
            </label>
            <button
              onClick={handleDemo}
              disabled={isProcessing}
              className="btn-primary flex items-center gap-1.5 px-4 py-1.5 text-[12px] disabled:opacity-50"
            >
              <Play className="h-3 w-3" />
              {isProcessing ? "Processing..." : "Run Demo"}
            </button>
          </div>
        </div>
      </motion.nav>

      {/* ═══════════════════════════════════
          HERO / TITLE EXPERIENCE
          ═══════════════════════════════════ */}
      <div
        ref={heroRef}
        onMouseMove={handleHeroMouse}
        className="relative overflow-hidden"
      >
        <div className="absolute inset-0 hero-atmosphere" />

        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{ x: heroGlowX, y: heroGlowY }}
        >
          <div className="absolute top-1/4 left-1/3 w-[500px] h-[400px] rounded-full bg-[#C75B6E] opacity-[0.05] blur-[120px] glow-drift" />
          <div className="absolute top-1/3 right-1/4 w-[400px] h-[350px] rounded-full bg-[#A04458] opacity-[0.04] blur-[100px] glow-drift" style={{ animationDelay: "2s" }} />
          <div className="absolute bottom-1/4 left-1/2 w-[350px] h-[300px] rounded-full bg-[#D98090] opacity-[0.03] blur-[90px] glow-drift" style={{ animationDelay: "1s" }} />
        </motion.div>

        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{ x: neuralX, y: neuralY }}
        >
          <NeuralBackground />
        </motion.div>

        <div className="absolute inset-0 neural-grid pointer-events-none" />

        <div className="relative z-10 mx-auto max-w-[1320px] px-5 md:px-8 xl:px-10">
          {/* Hero nav */}
          <motion.nav
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease }}
            className="flex items-center justify-between pt-6 pb-0 md:pt-8"
          >
            <div className="flex items-center gap-2.5">
              <ArchAngelLogo size={32} />
              <span className="text-[15px] font-semibold text-[#F5F7FA] tracking-tight">
                ArchAngel
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <label className="btn-secondary flex items-center gap-1.5 px-4 py-2 text-[13px] cursor-pointer">
                <Upload className="h-3.5 w-3.5" />
                Upload Clip
                <input type="file" accept="video/*" className="hidden" onChange={handleFileInput} />
              </label>
              <button
                onClick={handleDemo}
                disabled={isProcessing}
                className="btn-primary flex items-center gap-1.5 px-5 py-2 text-[13px] disabled:opacity-50"
              >
                <Play className="h-3.5 w-3.5" />
                {isProcessing ? "Processing..." : "Run Demo"}
              </button>
            </div>
          </motion.nav>

          {/* Hero headline */}
          <div className="flex flex-col items-center text-center pt-20 pb-24 md:pt-28 md:pb-32 lg:pt-32 lg:pb-36">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.1, ease }}
              className="mb-6"
            >
              <ArchAngelLogo size={56} className="mx-auto" />
            </motion.div>

            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2, ease }}
              className="section-label mb-5 tracking-[0.18em]"
            >
              Live Brain Activity Interface
            </motion.p>

            <motion.div style={{ x: textX, y: textY }}>
              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3, ease }}
                className="text-5xl md:text-6xl lg:text-7xl font-bold text-[#F5F7FA] tracking-tight leading-[1.08] max-w-3xl"
              >
                See cognition unfold
                <br />
                <span className="bg-gradient-to-r from-[#C75B6E] via-[#D98090] to-[#E8A0AE] bg-clip-text text-transparent">
                  in real time.
                </span>
              </motion.h1>
            </motion.div>

            <motion.p
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.45, ease }}
              className="text-base md:text-lg text-[rgba(245,247,250,0.52)] mt-6 max-w-lg leading-relaxed"
            >
              Upload a video or start a live feed — ArchAngel predicts and
              visualizes brain activity with precision.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.55, ease }}
              className="flex items-center gap-3 mt-10"
            >
              <button
                onClick={handleDemo}
                disabled={isProcessing}
                className="btn-primary flex items-center gap-2 px-7 py-3 text-[14px] disabled:opacity-50"
              >
                <Play className="h-4 w-4" />
                {isProcessing ? "Processing..." : "Run Demo"}
              </button>
              <label className="btn-secondary flex items-center gap-2 px-6 py-3 text-[14px] cursor-pointer">
                <Upload className="h-4 w-4" />
                Upload Clip
                <input type="file" accept="video/*" className="hidden" onChange={handleFileInput} />
              </label>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2, duration: 0.8 }}
              className="mt-16 flex flex-col items-center gap-2"
            >
              <span className="text-[11px] text-[rgba(245,247,250,0.24)] tracking-wider uppercase">
                Explore
              </span>
              <motion.div
                animate={{ y: [0, 6, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                <ArrowRight className="h-4 w-4 text-[rgba(245,247,250,0.2)] rotate-90" />
              </motion.div>
            </motion.div>
          </div>
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#111315] to-transparent pointer-events-none" />
      </div>

      {/* ═══════════════════════════════════
          MAIN CONTENT
          ═══════════════════════════════════ */}
      <div className="mx-auto max-w-[1320px] px-5 md:px-8 xl:px-10">

        {/* ═══════════ THREE-PANEL SHOWCASE ═══════════ */}
        <motion.section
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2, ease }}
          className="grid grid-cols-1 lg:grid-cols-[1fr_1.5fr_0.55fr] gap-4"
        >
          <VideoUploader
            onUpload={handleUpload}
            isProcessing={isProcessing}
            previewUrl={displayVideoUrl}
            videoRef={inputVideoRef}
            externalFileName={userFileName ?? undefined}
          />

          {prediction ? (
            <BrainViewer
              brainMovieUrl={prediction.brainMovieUrl}
              meanBrainUrl={prediction.meanBrainUrl}
              peakBrainUrl={prediction.peakBrainUrl}
              videoRef={brainVideoRef}
            />
          ) : isProcessing ? (
            <div className="card-surface flex items-center justify-center min-h-[280px]">
              <div className="flex flex-col items-center gap-3">
                <div className="w-6 h-6 spinner" />
                <span className="text-[13px] text-[rgba(245,247,250,0.48)]">{processingStage || "Analyzing neural patterns"}...</span>
              </div>
            </div>
          ) : (
            <Skeleton className="min-h-[280px]" />
          )}

          {prediction ? (
            <SnapshotRail
              totalActivity={prediction.totalActivity}
              alertLevel={prediction.alertLevel}
              alertLabel={prediction.alertLabel}
              dominantSystem={prediction.dominantSystem}
              engagement={prediction.engagement}
            />
          ) : (
            <Skeleton className="min-h-[280px]" />
          )}
        </motion.section>

        {/* ═══════════ ANALYTICS ═══════════ */}
        {isProcessing && !prediction ? (
          <div className="mt-4 grid grid-cols-1 lg:grid-cols-[1.6fr_1fr] gap-4">
            <Skeleton className="h-[380px]" />
            <div className="flex flex-col gap-4">
              <Skeleton className="h-[180px]" />
              <Skeleton className="h-[180px]" />
            </div>
          </div>
        ) : prediction ? (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3, ease }}
            className="mt-4 grid grid-cols-1 lg:grid-cols-[1.6fr_1fr] gap-4"
          >
            <ActivityChart timeline={prediction.timeline} systems={systemNames} />
            <div className="flex flex-col gap-4">
              <SystemCapsules scores={prediction.systemScores} />
              <RegionList regions={prediction.topRegions} />
            </div>
          </motion.section>
        ) : null}

        {/* ═══════════ SUMMARY ═══════════ */}
        {prediction && (
          <section className="mt-4">
            <SummaryPanel summary={prediction.summary} />
          </section>
        )}

        {/* ═══════════ TRUST STRIP ═══════════ */}
        <motion.section
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5, ease }}
          className="mt-10 mb-6 grid grid-cols-1 sm:grid-cols-3 gap-4"
        >
          {[
            { icon: Eye, title: "Live Input", desc: "Stream or upload video — analysis begins instantly.", accent: "#C75B6E" },
            { icon: Cpu, title: "Neural Visualization", desc: "Watch predicted brain activation unfold frame by frame.", accent: "#D98090" },
            { icon: FileText, title: "Interpretable Output", desc: "Region scores, system signals, and plain-language summaries.", accent: "#A04458" },
          ].map((item) => (
            <div key={item.title} className="card-surface px-5 py-5 flex items-start gap-4">
              <div
                className="flex h-10 w-10 items-center justify-center rounded-xl shrink-0"
                style={{ background: `${item.accent}15` }}
              >
                <item.icon className="h-[18px] w-[18px]" style={{ color: item.accent }} />
              </div>
              <div>
                <h4 className="text-[14px] font-semibold text-[#F5F7FA]">{item.title}</h4>
                <p className="text-[12px] text-[rgba(245,247,250,0.48)] mt-1 leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </motion.section>

        {/* ═══════════ FOOTER ═══════════ */}
        <footer className="py-8 text-center border-t border-[rgba(255,255,255,0.06)]">
          <div className="flex items-center justify-center gap-2.5">
            <ArchAngelLogo size={18} />
            <span className="text-[11px] text-[rgba(245,247,250,0.34)] font-medium tracking-wide">
              ArchAngel
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
