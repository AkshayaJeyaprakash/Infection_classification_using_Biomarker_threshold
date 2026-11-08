"""Shared inference for the saved UAB panel RF and neural network models."""

import numpy as np
import pandas as pd

from train_uab_panel_models import CLASSES, MARKERS


def predict_uab_panel(biomarker_values, artifact):
    """Predict one visit from ng/mL readings; omitted values remain missing."""
    unknown = set(biomarker_values) - set(MARKERS)
    if unknown:
        raise ValueError(f"Unknown UAB biomarkers: {', '.join(sorted(unknown))}")

    panel = pd.DataFrame(
        [{marker: biomarker_values.get(marker, np.nan) for marker in MARKERS}],
        columns=MARKERS,
        dtype=float,
    )
    readings = panel.to_numpy()
    if panel.isna().all(axis=None):
        raise ValueError("Enter at least one biomarker reading.")
    if np.isinf(readings).any() or (readings < 0).any():
        raise ValueError("Biomarker readings must be finite, non-negative numbers.")

    features = artifact["feature_engineer"].transform(panel)
    if not np.isfinite(features).all():
        raise ValueError("The biomarker panel could not be transformed into valid features.")

    model = artifact["model"]
    classes = list(model.classes_)
    if len(classes) != len(CLASSES) or set(classes) != set(CLASSES):
        raise ValueError("The UAB model does not contain the expected CORE-5 classes.")
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if (
        probabilities.shape != (1, len(classes))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError("The UAB model returned invalid class probabilities.")

    ranked_indices = np.argsort(-probabilities[0], kind="stable")
    classifications = [
        {
            "Rank": rank,
            "Class": str(classes[index]),
            "Confidence": float(probabilities[0, index] * 100),
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]
    return {
        "predicted_infection_type": classifications[0]["Class"],
        "confidence": classifications[0]["Confidence"],
        "classifications": classifications,
        "missing_biomarkers": [marker for marker in MARKERS if panel[marker].isna().iloc[0]],
    }
