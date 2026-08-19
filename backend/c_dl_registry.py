"""The C_DL architecture names and descriptions, with no torch import.

Split out from ``c_dl_pipeline`` so the model *catalogue* can be read without pulling in torch.
``model_reference.model_pool()`` needs the names on any machine — including the Streamlit
interpreter and a CI box without torch — and a family whose models vanish from the reference
because a library is missing is exactly the failure ``model_pool`` already guards against for
xgboost, lightgbm and catboost.

``c_dl_pipeline`` imports these names rather than defining its own, so there is one source of truth
and ``make_model`` cannot drift from what the page advertises.
"""
from __future__ import annotations

from typing import Dict

#: ``{NAME: description}``. Keys are UPPERCASE to match how the family labels itself in
#: ``SUMMARY.json`` and ``predictions_long.csv`` ("MLP", "TRANSFORMER"); ``make_model``
#: lowercases before dispatching.
C_DL_MODELS: Dict[str, str] = {
    "LSTM": "A recurrent network with gated memory, reading the sequence in order and carrying "
            "state forward. The standard sequence baseline.",
    "GRU": "A recurrent network like LSTM with a simpler gating scheme -- fewer parameters, "
           "often comparable accuracy on short series.",
    "DCNN": "A dilated causal convolution stack: each layer looks further back than the last, so "
            "a wide receptive field is reached without recurrence. Causal by construction.",
    "TRANSFORMER": "Self-attention over the input window, so any position can attend to any "
                   "earlier one directly rather than through carried state.",
    "MLP": "A plain feed-forward network over the flattened window. No sequence structure at all, "
           "which makes it the honest floor the sequence models have to beat.",
}


def registry_models() -> Dict[str, str]:
    """The models this family offers. Mirrors the A_STAT and E_QUANTILE registries."""
    return dict(C_DL_MODELS)
