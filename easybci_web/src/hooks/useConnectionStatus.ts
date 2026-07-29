import { useEffect, useRef, useState, useCallback } from "react";
import { warn } from "@/lib/debug";
import { apiUrl } from "@/lib/api";

export type ConnectionState = "checking" | "both_ok" | "dashboard_only" | "gateway_only" | "both_down";

interface ConnectionStatus {
  state: ConnectionState;
  dashboardOk: boolean;
  gatewayOk: boolean;
  retry: () => void;
  /** True after the gateway transitioned down→up — in-flight runs were lost
   *  and the user should re-send. Cleared via `acknowledgeGatewayRestart` (B15). */
  gatewayRestarted: boolean;
  acknowledgeGatewayRestart: () => void;
}

async function checkDashboard(): Promise<boolean> {
  try {
    const res = await fetch(apiUrl("/api/status"), { signal: AbortSignal.timeout(5000) });
    return res.ok;
  } catch (err) {
    warn("health", "dashboard probe failed", err);
    return false;
  }
}

async function checkGateway(): Promise<boolean> {
  try {
    const res = await fetch("/v1/capabilities", { signal: AbortSignal.timeout(5000) });
    return res.ok;
  } catch {
    try {
      const res = await fetch("/v1/health", { signal: AbortSignal.timeout(3000) });
      return res.ok;
    } catch (err) {
      warn("health", "gateway probe failed (capabilities + health)", err);
      return false;
    }
  }
}

// Adaptive intervals (ms): healthy → relaxed; first failure → fast probe;
// repeated failures → exponential backoff capped at 30s.
const HEALTHY_INTERVAL = 60_000;
const FAILURE_BASE = 5_000;
const FAILURE_MAX = 30_000;

function nextDelay(prevDelay: number, ok: boolean): number {
  if (ok) return HEALTHY_INTERVAL;
  if (prevDelay < FAILURE_BASE) return FAILURE_BASE;
  return Math.min(FAILURE_MAX, prevDelay * 2);
}

export function useConnectionStatus(): ConnectionStatus {
  const [dashboardOk, setDashboardOk] = useState(false);
  const [gatewayOk, setGatewayOk] = useState(false);
  const [state, setState] = useState<ConnectionState>("checking");
  const [gatewayRestarted, setGatewayRestarted] = useState(false);

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef(false);
  const delayRef = useRef(FAILURE_BASE);
  // Gateway reachability history for restart detection (B15): we only flag a
  // restart on a genuine up→down→up cycle, never on the initial cold start
  // (where the gateway may simply not be running yet).
  const gatewaySeenUpRef = useRef(false);
  const gatewaySeenDownRef = useRef(false);
  // Stable check function — avoids stale-closure resubscription on every render.
  const checkRef = useRef<() => Promise<void>>(async () => {});

  const scheduleNext = useCallback((delay: number) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      checkRef.current();
    }, delay);
  }, []);

  const check = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const [d, g] = await Promise.all([checkDashboard(), checkGateway()]);
      setDashboardOk(d);
      setGatewayOk(g);

      // Restart detection: the gateway was up, then went down, now it's back.
      // In-flight runs from before the restart are gone server-side, so the
      // user must re-send (B15).
      if (g) {
        if (gatewaySeenUpRef.current && gatewaySeenDownRef.current) {
          setGatewayRestarted(true);
          gatewaySeenDownRef.current = false;
        }
        gatewaySeenUpRef.current = true;
      } else if (gatewaySeenUpRef.current) {
        gatewaySeenDownRef.current = true;
      }

      const ok = d && g;
      if (ok) setState("both_ok");
      else if (d) setState("dashboard_only");
      else if (g) setState("gateway_only");
      else setState("both_down");
      delayRef.current = nextDelay(delayRef.current, ok);
      scheduleNext(delayRef.current);
    } finally {
      inFlightRef.current = false;
    }
  }, [scheduleNext]);

  const acknowledgeGatewayRestart = useCallback(() => setGatewayRestarted(false), []);

  // Keep ref pointing at the latest check function so timers and event listeners
  // always invoke a fresh closure.
  useEffect(() => {
    checkRef.current = check;
  }, [check]);

  const retry = useCallback(() => {
    delayRef.current = FAILURE_BASE;
    check();
  }, [check]);

  useEffect(() => {
    check();

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        // Tab became visible — probe immediately rather than wait for next tick.
        delayRef.current = FAILURE_BASE;
        check();
      }
    };
    const onOnline = () => {
      delayRef.current = FAILURE_BASE;
      check();
    };
    const onOffline = () => {
      // Browser knows the network is down — flip state immediately.
      setDashboardOk(false);
      setGatewayOk(false);
      setState("both_down");
    };

    document.addEventListener("visibilitychange", onVisibilityChange);
    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOffline);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOffline);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { state, dashboardOk, gatewayOk, retry, gatewayRestarted, acknowledgeGatewayRestart };
}
