from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import time


@dataclass
class StreamDeviceState:
    device_id: str
    patient_id: str
    sampling_rate: float
    samples: deque[float] = field(default_factory=deque)
    timestamps_ms: deque[int] = field(default_factory=deque)
    lead_off: bool = False
    last_seen_ms: int = 0
    total_samples_received: int = 0


class LiveStreamBuffer:
    def __init__(self, max_seconds: float = 120.0):
        self.max_seconds = max_seconds
        self._lock = Lock()
        self._devices: dict[str, StreamDeviceState] = {}

    @staticmethod
    def _now_ms() -> int:
        return int(time() * 1000)

    def ingest(
        self,
        *,
        device_id: str,
        patient_id: str,
        samples: list[float],
        sampling_rate: float,
        start_timestamp_ms: int | None,
        lead_off: bool,
    ) -> StreamDeviceState:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if not samples:
            raise ValueError("samples must not be empty.")

        with self._lock:
            state = self._devices.get(device_id)
            if state is None:
                state = StreamDeviceState(
                    device_id=device_id,
                    patient_id=patient_id,
                    sampling_rate=sampling_rate,
                )
                self._devices[device_id] = state

            state.patient_id = patient_id
            state.sampling_rate = sampling_rate
            state.lead_off = lead_off

            base_ts = start_timestamp_ms or self._now_ms()
            step_ms = max(1, int(round(1000.0 / sampling_rate)))
            for index, sample in enumerate(samples):
                state.samples.append(float(sample))
                state.timestamps_ms.append(base_ts + index * step_ms)

            state.total_samples_received += len(samples)
            state.last_seen_ms = state.timestamps_ms[-1]

            max_samples = max(1, int(self.max_seconds * sampling_rate))
            while len(state.samples) > max_samples:
                state.samples.popleft()
                state.timestamps_ms.popleft()

            return state

    def get_device(self, device_id: str) -> StreamDeviceState | None:
        with self._lock:
            state = self._devices.get(device_id)
            if state is None:
                return None
            return StreamDeviceState(
                device_id=state.device_id,
                patient_id=state.patient_id,
                sampling_rate=state.sampling_rate,
                samples=deque(state.samples),
                timestamps_ms=deque(state.timestamps_ms),
                lead_off=state.lead_off,
                last_seen_ms=state.last_seen_ms,
                total_samples_received=state.total_samples_received,
            )

    def get_recent_window(self, device_id: str, window_sec: float) -> dict:
        state = self.get_device(device_id)
        if state is None:
            raise KeyError(f"Unknown device_id: {device_id}")

        if not state.samples:
            return {
                "device_id": device_id,
                "patient_id": state.patient_id,
                "sampling_rate": state.sampling_rate,
                "samples": [],
                "timestamps_ms": [],
                "duration_sec": 0.0,
                "lead_off": state.lead_off,
                "last_seen_ms": state.last_seen_ms,
                "total_samples_received": state.total_samples_received,
            }

        max_samples = max(1, int(window_sec * state.sampling_rate))
        samples = list(state.samples)[-max_samples:]
        timestamps_ms = list(state.timestamps_ms)[-max_samples:]
        duration_sec = len(samples) / state.sampling_rate if state.sampling_rate else 0.0

        return {
            "device_id": device_id,
            "patient_id": state.patient_id,
            "sampling_rate": state.sampling_rate,
            "samples": samples,
            "timestamps_ms": timestamps_ms,
            "duration_sec": round(duration_sec, 3),
            "lead_off": state.lead_off,
            "last_seen_ms": state.last_seen_ms,
            "total_samples_received": state.total_samples_received,
        }

    def get_status(self, device_id: str) -> dict:
        state = self.get_device(device_id)
        if state is None:
            raise KeyError(f"Unknown device_id: {device_id}")

        last_seen_delta_ms = max(0, self._now_ms() - state.last_seen_ms) if state.last_seen_ms else None
        buffered_seconds = len(state.samples) / state.sampling_rate if state.sampling_rate else 0.0

        return {
            "device_id": state.device_id,
            "patient_id": state.patient_id,
            "sampling_rate": state.sampling_rate,
            "buffered_samples": len(state.samples),
            "buffered_seconds": round(buffered_seconds, 2),
            "lead_off": state.lead_off,
            "last_seen_ms": state.last_seen_ms,
            "last_seen_delta_ms": last_seen_delta_ms,
            "total_samples_received": state.total_samples_received,
            "is_streaming": bool(last_seen_delta_ms is not None and last_seen_delta_ms < 5000),
        }


stream_buffer = LiveStreamBuffer()
