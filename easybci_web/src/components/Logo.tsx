interface Props {
  size?: number;
  className?: string;
}

export function Logo({ size = 24, className }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
    >
      {/* Brain outline */}
      <path
        d="M16 4C10.5 4 6 8.5 6 14c0 3.5 1.8 6.5 4.5 8.2V26a2 2 0 002 2h7a2 2 0 002-2v-3.8C24.2 20.5 26 17.5 26 14c0-5.5-4.5-10-10-10z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        opacity="0.3"
        fill="currentColor"
        fillOpacity="0.05"
      />
      {/* EEG waveform across the brain */}
      <path
        d="M8 16h3l1.5-4 2 8 2-6 1.5 4 1.5-3 1 2H24"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Neural nodes */}
      <circle cx="11" cy="10" r="1.2" fill="currentColor" opacity="0.5" />
      <circle cx="16" cy="8" r="1.2" fill="currentColor" opacity="0.5" />
      <circle cx="21" cy="10" r="1.2" fill="currentColor" opacity="0.5" />
      {/* Connection lines between nodes */}
      <path
        d="M11 10l5-2 5 2"
        stroke="currentColor"
        strokeWidth="0.8"
        opacity="0.3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function LogoMark({ size = 16, className }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      className={className}
    >
      <path
        d="M2 8h2l1-3 1.5 6 1.5-4.5 1 3 1-2 .5 1H14"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
