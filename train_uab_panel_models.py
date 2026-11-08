"""Feature-engineering classes required by the saved UAB panel artifacts.

Kept under the original training module name for pickle compatibility.
The feature definitions below match the supplied training script.
"""


from dataclasses import dataclass, field

import numpy as np


CLASSES = ["BR", "BVR", "C", "N", "VR"]


MARKERS = ["IL-6", "IL-27", "IP-10", "CRP", "D-Dimer", "PCT", "Resistin"]


BACTERIAL_MARKERS = {"CRP", "PCT"}


VIRAL_MARKERS = {"IP-10"}


RATIO_PAIRS = [("PCT", "IP-10"), ("CRP", "IP-10"), ("IL-6", "IP-10")]


IQR_FLOOR = 0.15


CLIP = 10.0


@dataclass
class MarkerStats:
    median_log: dict = field(default_factory=dict)
    q1_log: dict = field(default_factory=dict)
    q3_log: dict = field(default_factory=dict)
    global_median_raw: dict = field(default_factory=dict)
    class_center: dict = field(default_factory=dict)         # (marker, class) -> mean rel score
    global_class_center: dict = field(default_factory=dict)


class UABPanelFeatureEngineer:
    """
    Fits all training-only statistics once (medians, IQRs, per-class
    centers -- all from the training visits only), then transforms any
    dataframe (training fold, held-out visit, or new incoming visit) into
    the literature-informed feature matrix described at the top of this
    file.
    """

    def __init__(self):
        self.stats_ = MarkerStats()

    def fit(self, df, y):
        df = df.copy()
        df["label"] = y

        for m in MARKERS:
            self.stats_.global_median_raw[m] = float(df[m].median(skipna=True))

        for m in MARKERS:
            logv = np.log1p(df[m].fillna(self.stats_.global_median_raw[m]))
            self.stats_.median_log[m] = float(logv.median())
            self.stats_.q1_log[m] = float(logv.quantile(0.25))
            self.stats_.q3_log[m] = float(logv.quantile(0.75))

        rel = self._relative_scores(df)
        for m in MARKERS:
            df[f"_rel_{m}"] = rel[m]
            vals = df[f"_rel_{m}"]
            self.stats_.global_class_center[m] = float(vals.mean())
            for cls in CLASSES:
                sub = df.loc[df["label"] == cls, f"_rel_{m}"]
                if len(sub) >= 2:
                    self.stats_.class_center[(m, cls)] = float(sub.mean())
        return self

    def _relative_scores(self, df):
        out = {}
        for m in MARKERS:
            filled = df[m].fillna(self.stats_.global_median_raw[m]).to_numpy(dtype=float)
            logv = np.log1p(filled)
            med = self.stats_.median_log[m]
            iqr = max(self.stats_.q3_log[m] - self.stats_.q1_log[m], IQR_FLOOR)
            out[m] = np.clip((logv - med) / iqr, -CLIP, CLIP)
        return out

    def _center(self, m, cls):
        return self.stats_.class_center.get((m, cls), self.stats_.global_class_center[m])

    def transform(self, df):
        df = df.copy()
        n = len(df)
        rel = self._relative_scores(df)

        cols = {}
        missing_flags = {}
        for m in MARKERS:
            cols[f"rel_{m}"] = rel[m]
            missing_flags[m] = df[m].isna().astype(float).to_numpy()
            cols[f"missing_{m}"] = missing_flags[m]

        # literature direction-prior (bacterial +1 / viral -1), only for
        # markers the literature review actually covers
        for m in BACTERIAL_MARKERS:
            cols[f"dir_{m}"] = rel[m]
        for m in VIRAL_MARKERS:
            cols[f"dir_{m}"] = -rel[m]

        # elevation-over-control and bacterial-viral axis, every marker
        for m in MARKERS:
            c_control = self._center(m, "C")
            cols[f"elev_{m}"] = rel[m] - c_control
            cols[f"elev_pos_{m}"] = np.maximum(rel[m] - c_control, 0.0)

            c_br = self._center(m, "BR")
            c_vr = self._center(m, "VR")
            sign = 1.0 if (c_br - c_vr) >= 0 else -1.0
            cols[f"bvaxis_{m}"] = sign * (rel[m] - 0.5 * (c_br + c_vr))

        # Stage-5 rule-selected ratios (log-space difference)
        for a, b in RATIO_PAIRS:
            la = np.log1p(df[a].fillna(self.stats_.global_median_raw[a]).to_numpy(dtype=float))
            lb = np.log1p(df[b].fillna(self.stats_.global_median_raw[b]).to_numpy(dtype=float))
            cols[f"ratio_{a}_{b}"] = la - lb

        return np.column_stack([cols[k] for k in sorted(cols.keys())])
