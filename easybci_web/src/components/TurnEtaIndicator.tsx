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
  // The clock lives in state and ticks every 500ms so the countdown is
  // visibly live while render itself stays pure.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!latestTurnEta) return;
    const tick = () => setNow(Date.now());
    const first = setTimeout(tick, 0);
    const id = setInterval(tick, 500);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, [latestTurnEta]);

  const remaining = computeRemainingSeconds(latestTurnEta, now);
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
