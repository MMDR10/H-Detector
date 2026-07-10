"""
Jailbreak and anomaly detection pipeline using H metric.

The detector analyzes LLM hidden states to identify:
1. Jailbreak attempts (H spike in storm zone)
2. Quiet jailbreaks (anomalous but sub-threshold H patterns)
3. Role confusion (style-driven role perception shifts)
4. Semantic anomalies (abstraction vs literal processing)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .helicity import compute_helicity, compute_layer_h, compute_storm_zone_intensity


@dataclass
class DetectionResult:
    """Result from the H detection pipeline."""

    is_jailbreak: bool
    confidence: float  # 0.0 to 1.0
    h_overall: float
    storm_mean: float
    storm_peak: float
    storm_baseline_ratio: float
    mechanism: Optional[str] = None  # "bistability_trap", "refusal_collapse", etc.
    layer_h: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        mech = f" [{self.mechanism}]" if self.mechanism else ""
        return (
            f"{'🚨 JAILBREAK' if self.is_jailbreak else '✅ BENIGN'}{mech}\n"
            f"  H={self.h_overall:.3f}, Storm peak={self.storm_peak:.3f}, "
            f"Storm mean={self.storm_mean:.3f}, Ratio={self.storm_baseline_ratio:.3f}"
        )


# Thresholds calibrated on Qwen2.5 0.5B–14B experiments
# These are empirical starting points; adjust for your model.
THRESHOLDS = {
    "h_alert": 2.5,           # Overall H above this → suspicious
    "storm_peak_alert": 3.5,  # Storm zone peak above this → suspicious
    "storm_ratio_alert": 1.8, # Storm-to-baseline ratio above this → suspicious
    "h_jailbreak": 3.5,       # Overall H above this → high confidence jailbreak
    "storm_peak_jailbreak": 5.0,  # Storm peak above this → high confidence
}


class HDetector:
    """
    Geometric jailbreak/anomaly detector using the H metric.

    Usage:
        detector = HDetector()
        result = detector.detect(hidden_states)
        print(result.summary())
    """

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = {**THRESHOLDS, **(thresholds or {})}
        self._h_history: list[float] = []

    def detect(
        self,
        hidden_states: np.ndarray,
        prompt_text: Optional[str] = None,
    ) -> DetectionResult:
        """
        Run full detection pipeline on hidden state activations.

        Parameters
        ----------
        hidden_states : np.ndarray
            Shape (n_layers, hidden_dim). Activations from each transformer layer.
        prompt_text : str, optional
            The input prompt (for logging/debugging only).

        Returns
        -------
        DetectionResult
        """
        # Compute H metrics
        h_overall = compute_helicity(hidden_states)
        layer_h = compute_layer_h(hidden_states)
        storm_mean, storm_peak, storm_ratio = compute_storm_zone_intensity(layer_h)

        self._h_history.append(h_overall)

        # Determine mechanism
        mechanism = self._classify_mechanism(layer_h, h_overall, storm_peak)

        # Determine jailbreak status
        is_jailbreak, confidence = self._classify_threat(
            h_overall, storm_mean, storm_peak, storm_ratio
        )

        return DetectionResult(
            is_jailbreak=is_jailbreak,
            confidence=confidence,
            h_overall=h_overall,
            storm_mean=storm_mean,
            storm_peak=storm_peak,
            storm_baseline_ratio=storm_ratio,
            mechanism=mechanism,
            layer_h=layer_h,
        )

    def _classify_mechanism(
        self,
        layer_h: np.ndarray,
        h_overall: float,
        storm_peak: float,
    ) -> str:
        """
        Classify the jailbreak mechanism based on H profile shape.
        """
        if len(layer_h) < 25:
            return "unknown"

        # Mechanism 1: Δ Bistability Trap
        # High storm zone (L15-21) + sharp drop at L27
        storm_zone = layer_h[15:21] if len(layer_h) > 21 else layer_h[-6:-1]
        refusal_layer = layer_h[-1] if len(layer_h) >= 27 else layer_h[-1]

        if storm_peak > 3.0 and refusal_layer < np.mean(storm_zone) * 0.7:
            return "bistability_trap"

        # Mechanism 2: Refusal Collapse
        # Low overall H but refusal layer signature absent
        if h_overall < 1.8 and storm_peak < 2.0:
            return "refusal_collapse"

        # Mechanism 3: Quiet Jailbreak
        # H in normal range but storm zone shows anomalous structure
        if 1.5 < h_overall < 2.5 and storm_peak > 3.0:
            return "quiet_jailbreak"

        return "standard"

    def _classify_threat(
        self,
        h_overall: float,
        storm_mean: float,
        storm_peak: float,
        storm_ratio: float,
    ) -> tuple[bool, float]:
        """
        Classify whether the input is a jailbreak attempt.

        Returns (is_jailbreak, confidence).
        """
        t = self.thresholds

        # High confidence jailbreak signals
        if h_overall > t["h_jailbreak"] or storm_peak > t["storm_peak_jailbreak"]:
            return True, 0.85

        # Alert zone
        alerts = 0
        if h_overall > t["h_alert"]:
            alerts += 1
        if storm_peak > t["storm_peak_alert"]:
            alerts += 1
        if storm_ratio > t["storm_ratio_alert"]:
            alerts += 1

        if alerts >= 2:
            return True, 0.65
        elif alerts == 1:
            return True, 0.50  # borderline

        # Benign
        return False, 1.0 - min(1.0, (h_overall - 1.0) / 2.0)
