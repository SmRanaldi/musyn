"""
musyn — Muscle Synergy Analysis from Surface EMG

Three algorithms implemented from peer-reviewed research:

1. Adaptive envelope extraction — Ranaldi et al. (2018)
2. NMF synergy extraction — Soomro et al. (2018)
3. AIC-based synergy number selection — Ranaldi et al. (2021)

Quick start
-----------
>>> import musyn
>>> envelope = musyn.extract_envelope(signal, fs=1000.0)
>>> W, C, info = musyn.extract_synergies(D, n_synergies=4)
>>> k = musyn.select_synergy_number(D)
"""

from musyn.envelope.api import extract_envelope
from musyn.decomposition.api import extract_synergies
from musyn.selection.api import select_synergy_number
from musyn.metrics.quality import quality_ratio, vaf, r_squared

try:
    from musyn._version import __version__
except ImportError:
    __version__ = "0.1.0"

__all__ = [
    "extract_envelope",
    "extract_synergies",
    "select_synergy_number",
    "quality_ratio",
    "vaf",
    "r_squared",
    "__version__",
]
