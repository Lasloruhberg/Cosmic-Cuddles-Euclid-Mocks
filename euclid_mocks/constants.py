"""Fixed instrument and simulation numbers.

These are properties of Euclid Q1 and of IllustrisTNG, not run choices. Run
choices live in ``config/default.yaml``.
"""

from enum import Enum

import numpy as np
from astropy.cosmology import FlatLambdaCDM


class EuclidBand(Enum):
    """Name a Euclid band; the value is the MER file token and the FITS EXTNAME."""

    VIS = "VIS"
    NIR_Y = "NIR-Y"
    NIR_J = "NIR-J"
    NIR_H = "NIR-H"

    @property
    def tng_filter(self) -> str:
        """Return the TNG visualisation API filter name of this band."""
        return _TNG_FILTER[self]

    @property
    def instrument_folder(self) -> str:
        """Return the MER sub-folder (``VIS`` or ``NISP``) holding this band."""
        return "VIS" if self is EuclidBand.VIS else "NISP"


_TNG_FILTER = {
    EuclidBand.VIS: "euclid_vis",
    EuclidBand.NIR_Y: "euclid_y",
    EuclidBand.NIR_J: "euclid_j",
    EuclidBand.NIR_H: "euclid_h",
}
NISP_BANDS = (EuclidBand.NIR_Y, EuclidBand.NIR_J, EuclidBand.NIR_H)
ALL_BANDS = (EuclidBand.VIS, *NISP_BANDS)

JANSKY_AB_ZEROPOINT = 3631.0

# --- IllustrisTNG (Planck 2015, Nelson et al. 2019a) -------------------------
TNG_H = 0.6774
TNG_COSMOLOGY = FlatLambdaCDM(H0=100.0 * TNG_H, Om0=0.3089, Ob0=0.0486, Tcmb0=2.7255)

# --- VIS (McCracken et al., VIS pipeline paper) -------------------------------
VIS_PIXEL_SCALE_ARCSEC = 0.1
VIS_NOMINAL_EXPOSURE_S = 566.0
VIS_SHORT_EXPOSURE_S = 95.0
# Only the first two dithers of a Q1 ROS carry an extra short VIS exposure.
VIS_N_SHORT_EXPOSURE_DITHERS = 2
VIS_GAIN_E_PER_ADU = 3.48
VIS_ZEROPOINT_ADU_PER_S = 24.57
VIS_ADU_SATURATION = 65535

# --- NISP (Jahnke et al. 2024; Polenta et al., NIR pipeline paper) ------------
NISP_PIXEL_SCALE_ARCSEC = 0.3
NISP_N_GROUPS = 4
NISP_N_FRAMES = 16
NISP_N_DROPPED = 4
NISP_FRAME_TIME_S = 1.45408
_NISP_GAIN_ADU_PER_E = np.array(
    [0.5208, 0.5102, 0.5319, 0.5128, 0.5291, 0.5236, 0.5348, 0.5376,
     0.5525, 0.5155, 0.5525, 0.5405, 0.5208, 0.5102, 0.5076, 0.5263]
)  # fmt: skip
_NISP_READ_NOISE_ADU = np.array(
    [4.8708, 4.4655, 4.7514, 3.9748, 4.9505, 4.784, 4.6474, 4.908,
     5.9996, 4.833, 4.9046, 4.806, 6.3976, 4.3443, 4.665, 5.1524]
)  # fmt: skip
NISP_GAIN_ADU_PER_E = float(np.average(_NISP_GAIN_ADU_PER_E))
NISP_READ_NOISE_E = float(np.average(_NISP_READ_NOISE_ADU / _NISP_GAIN_ADU_PER_E))
# Inverse sensitivity: micro-Jy per (e-/s); already folds in the detector QE.
NISP_INVERSE_SENSITIVITY = {
    EuclidBand.NIR_Y: 0.3938,
    EuclidBand.NIR_J: 0.309,
    EuclidBand.NIR_H: 0.335,
}
NISP_ZEROPOINT = {
    EuclidBand.NIR_Y: 29.8,
    EuclidBand.NIR_J: 30.0,
    EuclidBand.NIR_H: 29.9,
}
# Caps Poisson expectations so numpy never sees overflowing or invalid lambdas.
NISP_POISSON_LAMBDA_CAP = 65535.0

# --- Q1 reference observing sequence dither (Scaramella et al. 2022) ----------
# Negative because the detector grid, not the source, is moved when drizzling.
Q1_DITHER_PATTERN_ARCSEC = (
    (0.0, 0.0),
    (-50.0, -100.0),
    (-50.0, -200.0),
    (-100.0, -300.0),
)
