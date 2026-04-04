"use client";

export function ArchAngelLogo({ size = 32, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Halo / guardian arc */}
      <path
        d="M12 20 C12 4, 52 4, 52 20"
        stroke="url(#haloGrad)"
        strokeWidth="1.6"
        strokeLinecap="round"
        fill="none"
      />

      {/* Brain outer contour — side profile */}
      <path
        d="M20 42 C15 42, 13 37, 14 33 C14 29, 17 25, 19 23 C18.5 21, 17.5 18, 20 16 C23 14, 27 13, 31 13 C35 13, 39 14, 42 16.5 C45 19, 47 22, 46.5 26 C47 28, 47.5 31, 46 34 C44 37, 41 40, 38 41.5 C36 42, 34 42.5, 31 42.5"
        stroke="url(#brainGrad)"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />

      {/* Central sulcus */}
      <path
        d="M31 15 C30 20, 33 25, 30 30 C28 34, 31 38, 31 41"
        stroke="#C75B6E"
        strokeWidth="0.8"
        strokeLinecap="round"
        fill="none"
        opacity="0.3"
      />

      {/* Cortical folds */}
      <path d="M21 23 C24 21, 27.5 22, 28 25" stroke="#C75B6E" strokeWidth="0.7" strokeLinecap="round" fill="none" opacity="0.25" />
      <path d="M34 17 C37 16.5, 41 18.5, 41.5 22" stroke="#C75B6E" strokeWidth="0.7" strokeLinecap="round" fill="none" opacity="0.25" />
      <path d="M19 33 C22 31, 26 32.5, 27 35" stroke="#C75B6E" strokeWidth="0.7" strokeLinecap="round" fill="none" opacity="0.2" />

      {/* Neural pathway nodes */}
      <circle cx="24" cy="25" r="1.4" fill="#C75B6E" opacity="0.6" />
      <circle cx="37" cy="21" r="1.4" fill="#A04458" opacity="0.55" />
      <circle cx="31" cy="31" r="1.4" fill="#D98090" opacity="0.55" />
      <circle cx="41" cy="30" r="1.1" fill="#C75B6E" opacity="0.4" />
      <circle cx="22" cy="35" r="1.1" fill="#A04458" opacity="0.35" />

      {/* Neural connections */}
      <line x1="24" y1="25" x2="31" y2="31" stroke="#C75B6E" strokeWidth="0.5" opacity="0.2" />
      <line x1="37" y1="21" x2="31" y2="31" stroke="#A04458" strokeWidth="0.5" opacity="0.2" />
      <line x1="37" y1="21" x2="41" y2="30" stroke="#C75B6E" strokeWidth="0.5" opacity="0.18" />
      <line x1="24" y1="25" x2="37" y2="21" stroke="#D98090" strokeWidth="0.4" opacity="0.15" />
      <line x1="22" y1="35" x2="31" y2="31" stroke="#A04458" strokeWidth="0.4" opacity="0.15" />

      {/* Brain stem */}
      <path
        d="M28 42.5 C28 46, 29.5 48.5, 31 50 C32.5 48.5, 33.5 46, 33.5 42.5"
        stroke="#C75B6E"
        strokeWidth="1"
        strokeLinecap="round"
        fill="none"
        opacity="0.3"
      />

      <defs>
        <linearGradient id="haloGrad" x1="12" y1="12" x2="52" y2="12">
          <stop offset="0%" stopColor="#A04458" stopOpacity="0.5" />
          <stop offset="50%" stopColor="#C75B6E" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#D98090" stopOpacity="0.4" />
        </linearGradient>
        <linearGradient id="brainGrad" x1="14" y1="13" x2="47" y2="42">
          <stop offset="0%" stopColor="#D98090" />
          <stop offset="60%" stopColor="#C75B6E" />
          <stop offset="100%" stopColor="#A04458" />
        </linearGradient>
      </defs>
    </svg>
  );
}
