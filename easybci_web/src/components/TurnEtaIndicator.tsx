// Typing-dots-row ETA hint. During streaming shows ONLY the countdown
// (`est +Xs`). When the countdown reaches zero we keep displaying `est +0s`
// quietly — the next turn-scope ETA emit replaces it. No elapsed counter,
// no yellow "over" warning.
import { useEffect, useState } from "react";

import {
  computeRemainingSeconds,
  formatEtaText,
  type TurnEta,
} from "../lib/turnEta";

type Props = {
  /** Latest turn-scope ETA, or null while no estimate is active. */
  latestTurnEta: TurnEta | null;
};

export function TurnEtaIndicator({ latestTurnEta }: Props) {
  // Tick every 500ms so the countdown is visibly live.
  const [, forceTick] = useState(0);
  useEffect(() => {
    if (!latestTurnEta) return;
    const id = setInterval(() => forceTick((n) => (n + 1) % 1_000_000), 500);
    return () => clearInterval(id);
  }, [latestTurnEta]);

  const remaining = computeRemainingSeconds(latestTurnEta, Date.now());
  const etaText = formatEtaText(remaining);

  if (!etaText) return null;

  return (
    <span
      className="ml-2 inline-flex items-center text-[11px]"
      style={{ color: "var(--text-muted)" }}
    >
      est {etaText}
    </span>
  );
}
