"""Forward-only physiological preprocessing described in Section III-D."""

from __future__ import annotations

from collections import deque
from importlib.metadata import version

import numpy as np
from scipy import signal


class StatefulButterworth:
    def __init__(self, sampling_rate: int, channels: int, low: float, high: float) -> None:
        self.sampling_rate = sampling_rate
        self.sos = np.vstack((
            signal.butter(2, low, btype="highpass", fs=sampling_rate, output="sos"),
            signal.butter(2, high, btype="lowpass", fs=sampling_rate, output="sos")))
        self.channels = channels
        self.reset()

    def reset(self) -> None:
        self.zi = np.zeros((len(self.sos), 2, self.channels), dtype=np.float64)
        self.last_time = None

    def push(self, values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.channels)
        times = np.asarray(times, dtype=np.float64)
        valid = np.isfinite(values).all(axis=-1)
        output = np.zeros_like(values)
        for index, time in enumerate(times):
            if self.last_time is not None and time <= self.last_time:
                raise ValueError("Sensor timestamps must be strictly increasing")
            if (not valid[index] or self.last_time is not None
                    and time - self.last_time > 1.5 / self.sampling_rate):
                self.zi.fill(0)
            if valid[index]:
                filtered, self.zi = signal.sosfilt(self.sos, values[index:index + 1],
                                                   axis=0, zi=self.zi)
                output[index] = filtered[0]
            self.last_time = float(time)
        return output, valid


class CausalPhysiology:
    """Incremental EEG/PPG preprocessing; every call consumes only a prefix.

    push_eeg()/push_ppg() take new native samples. decision() releases one
    spectral observation and zero or more newly confirmed pulse-rate values.
    """

    BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))

    def __init__(self) -> None:
        self.eeg_filter = StatefulButterworth(200, 8, 1, 45)
        self.ppg_filter = StatefulButterworth(100, 1, 0.5, 8)
        self.eeg = deque(maxlen=801)
        self.ppg = deque(maxlen=1001)
        self.last_peak = None
        self.last_decision = -float("inf")

    def push_eeg(self, values, times) -> dict:
        times = np.asarray(times, dtype=np.float64)
        filtered, valid = self.eeg_filter.push(values, times)
        self.eeg.extend(zip(times, filtered, valid, strict=True))
        return {"values": filtered.astype(np.float32), "support": np.column_stack((times, times)),
                "available": times.copy(), "mask": valid}

    def push_ppg(self, values, times) -> None:
        times = np.asarray(times, dtype=np.float64)
        filtered, valid = self.ppg_filter.push(values, times)
        self.ppg.extend(zip(times, filtered[:, 0], valid, strict=True))

    @staticmethod
    def _trailing(buffer, now: float, seconds: float, sampling_rate: int):
        rows = [row for row in buffer if now - seconds < row[0] <= now]
        if len(rows) != round(seconds * sampling_rate) or not all(row[2] for row in rows):
            return None
        times = np.asarray([row[0] for row in rows], dtype=np.float64)
        if np.any(np.abs(np.diff(times) - 1 / sampling_rate) > 0.1 / sampling_rate):
            return None
        return times, np.asarray([row[1] for row in rows], dtype=np.float64)

    def decision(self, now: float) -> dict[str, list[dict]]:
        if now <= self.last_decision:
            raise ValueError("Physiological decisions must advance in time")
        self.last_decision = now
        result = {"spec": [], "hr": []}
        eeg = self._trailing(self.eeg, now, 4.0, 200)
        if eeg is not None:
            times, values = eeg
            frequencies, density = signal.welch(values, fs=200, window="hann", nperseg=400,
                                                noverlap=200, axis=0, detrend="constant")
            powers = []
            for index, (low, high) in enumerate(self.BANDS):
                band = (frequencies >= low) & (frequencies <= high if index == 4 else frequencies < high)
                powers.append(density[band].sum(axis=0) * (frequencies[1] - frequencies[0]))
            # Channel-major order: five log band powers for channel 1, then 2, ...
            features = np.log(np.maximum(np.stack(powers, axis=-1), 1e-12)).reshape(40)
            result["spec"].append({"values": features, "support": [now - 4.0, now], "available": now})
        ppg = self._trailing(self.ppg, now, 10.0, 100)
        if ppg is not None:
            if version("neurokit2") != "0.2.11":
                raise RuntimeError("The manuscript's pulse detector requires neurokit2==0.2.11")
            import neurokit2 as nk
            times, values = ppg
            try:
                candidates = nk.ppg_findpeaks(values, sampling_rate=100, method="elgendi")["PPG_Peaks"]
            except (ValueError, IndexError):
                # No detectable pulse in this observed window: release no value.
                candidates = []
            for index in sorted(candidates):
                peak = float(times[int(index)])
                if peak > now - 0.4 + 1e-9:
                    continue
                if self.last_peak is not None and peak - self.last_peak < 0.3 - 1e-9:
                    continue
                if self.last_peak is not None:
                    result["hr"].append({"values": [60.0 / (peak - self.last_peak)],
                        "support": [self.last_peak, peak], "available": now})
                self.last_peak = peak
        return result


def prepare_physiology(eeg, eeg_times, ppg, ppg_times, decisions) -> dict:
    """Replay a recording chronologically; filtering never uses future samples."""
    processor = CausalPhysiology()
    eeg_times, ppg_times = np.asarray(eeg_times), np.asarray(ppg_times)
    eeg, ppg = np.asarray(eeg), np.asarray(ppg)
    output = {name: [] for name in ("time", "spec", "hr")}
    eeg_cursor = ppg_cursor = 0
    for now in decisions:
        eeg_stop = int(np.searchsorted(eeg_times, now, side="right"))
        ppg_stop = int(np.searchsorted(ppg_times, now, side="right"))
        if eeg_stop > eeg_cursor:
            batch = processor.push_eeg(eeg[eeg_cursor:eeg_stop], eeg_times[eeg_cursor:eeg_stop])
            output["time"].extend({key: value[index] for key, value in batch.items()}
                                  for index in range(eeg_stop - eeg_cursor))
        if ppg_stop > ppg_cursor:
            processor.push_ppg(ppg[ppg_cursor:ppg_stop], ppg_times[ppg_cursor:ppg_stop])
        released = processor.decision(float(now))
        for name in ("spec", "hr"):
            output[name].extend(released[name])
        eeg_cursor, ppg_cursor = eeg_stop, ppg_stop
    packed = {}
    for name, width in (("time", 8), ("spec", 40), ("hr", 1)):
        rows = output[name]
        packed[name] = {
            "values": np.asarray([row["values"] for row in rows], dtype=np.float32).reshape(-1, width),
            "support": np.asarray([row["support"] for row in rows], dtype=np.float64).reshape(-1, 2),
            "available": np.asarray([row["available"] for row in rows], dtype=np.float64),
            "mask": np.asarray([row.get("mask", True) for row in rows], dtype=bool)}
    return packed
