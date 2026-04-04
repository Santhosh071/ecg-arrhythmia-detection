export function WaveformPreview() {
  return (
    <div className="waveform" aria-label="ECG waveform preview">
      <svg viewBox="0 0 800 240" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M0 120H70L95 110L108 120L121 40L140 190L162 120L238 120L260 112L272 120L285 58L304 178L325 120H390L415 107L428 120L442 68L461 176L482 120H560L584 115L597 120L610 74L628 166L650 120H800"
          stroke="#0f766e"
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M442 68L461 176L482 120"
          stroke="#b42318"
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}
