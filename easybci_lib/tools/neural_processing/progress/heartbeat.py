"""HeartbeatDaemon — background thread that keeps the heartbeat alive during
long blocking calls (LLM inference, subprocess) where the main thread cannot
itself call `tracker.tick()`."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from .tracker import StagedProgressTracker

logger = logging.getLogger(__name__)


class HeartbeatDaemon:
    """Owns a daemon thread that calls `tracker.tick()` at a fixed interval
    whenever the tracker has an active stage.

    Over-emitting at fixed interval is cheap — frontend deduplicates by
    (sub_step, sub_index); the only thing that changes is heartbeat_ts,
    which is exactly what we want during a blocking call.
    """

    def __init__(
        self,
        tracker: StagedProgressTracker,
        *,
        check_interval_s: float = 5.0,
    ) -> None:
        self._tracker = tracker
        self._check_interval_s = check_interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="easybci-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._check_interval_s * 3)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._tracker.active_stage is None:
                    self._stop_event.wait(self._check_interval_s)
                    continue
                self._tracker.tick()
            except Exception as exc:
                logger.debug("heartbeat daemon tick failed: %s", exc)
            self._stop_event.wait(self._check_interval_s)
