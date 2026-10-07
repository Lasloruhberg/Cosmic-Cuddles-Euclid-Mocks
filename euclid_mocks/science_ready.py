"""Assemble, validate and locate the final multi-band ScienceReady FITS files.

A ScienceReady file holds ``[Primary, VIS, NIR-Y, NIR-J, NIR-H, VISDET,
NIR-YDET, NIR-JDET, NIR-HDET]``: science images in Jy/pixel and
detection-significance maps, all on the VIS pixel grid.
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dotmap import DotMap

from euclid_mocks.constants import ALL_BANDS, JANSKY_AB_ZEROPOINT, EuclidBand


@dataclass
class BandProduct:
    """Collect everything the ScienceReady file needs from one band.

    Attributes:
        band (EuclidBand): Band.
        mock (np.ndarray): Noisy stack before PSF and background (for the DET map).
        science (np.ndarray): PSF-convolved stack plus background, mosaic units.
        header (fits.Header): Band header (VIS WCS, MAGZERO, mock and background cards).
        detection_level (float): Central signal over background noise.
        noise (float): Background standard deviation in mosaic units.
    """

    band: EuclidBand
    mock: np.ndarray
    science: np.ndarray
    header: fits.Header
    detection_level: float
    noise: float


def mosaic_to_jansky(image: np.ndarray, zeropoint: float) -> np.ndarray:
    """Convert MER mosaic units with AB zeropoint ``zeropoint`` to Jy/pixel.

    Args:
        image (np.ndarray): Image in mosaic units.
        zeropoint (float): AB zeropoint of those units.

    Returns:
        np.ndarray: Image in Jy/pixel.
    """
    return image * 10 ** (-0.4 * zeropoint) * JANSKY_AB_ZEROPOINT


def assert_same_geometry(reference: fits.ImageHDU, other: fits.ImageHDU) -> None:
    """Raise unless two HDUs have identical shape and celestial WCS.

    This is the final guard of the pipeline: every extension of a
    ScienceReady file must lie on the VIS reference grid.

    Args:
        reference (fits.ImageHDU): The VIS extension.
        other (fits.ImageHDU): Extension to compare.

    Raises:
        ValueError: On any difference in shape, CRPIX, CRVAL, pixel matrix or CTYPE.
    """
    if other.data.shape != reference.data.shape:
        raise ValueError(f"{other.name} shape {other.data.shape} != VIS {reference.data.shape}")
    ref, cur = WCS(reference.header), WCS(other.header)
    checks = {
        "CRPIX": np.allclose(ref.wcs.crpix, cur.wcs.crpix, atol=1e-6),
        "CRVAL": np.allclose(ref.wcs.crval, cur.wcs.crval, atol=1e-8),
        "pixel matrix": np.allclose(ref.pixel_scale_matrix, cur.pixel_scale_matrix, atol=1e-12),
        "CTYPE": tuple(ref.wcs.ctype) == tuple(cur.wcs.ctype),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{other.name} WCS differs from VIS in {', '.join(failed)}")


def build_science_ready(products: dict[EuclidBand, BandProduct]) -> fits.HDUList:
    """Assemble the ScienceReady HDU list from the four band products.

    Science extensions are converted to Jy/pixel with the band's final
    ``MAGZERO``. DET extensions divide the pre-PSF mock by the background
    noise and indicate where the source itself is significant. Every
    extension is checked against the VIS geometry before the list is returned.

    Args:
        products (dict[EuclidBand, BandProduct]): One product per band.

    Returns:
        fits.HDUList: ``[Primary, 4 science, 4 DET]`` extensions.
    """
    science, detection = [], []
    for band in ALL_BANDS:
        p = products[band]
        header = p.header.copy()
        header["BUNIT"] = "Jy"
        science.append(
            fits.ImageHDU(data=mosaic_to_jansky(p.science, header["MAGZERO"]), header=header, name=band.value)
        )
        det = p.mock / p.noise if p.noise > 0 else np.zeros_like(p.mock)
        det_header = p.header.copy()
        det_header["BUNIT"] = "sigma"
        detection.append(fits.ImageHDU(data=det, header=det_header, name=f"{band.value}DET"))

    hdul = fits.HDUList([fits.PrimaryHDU(), *science, *detection])
    for hdu in hdul[2:]:
        assert_same_geometry(hdul[1], hdu)
    return hdul


def science_ready_dir(cfg: DotMap) -> Path:
    """Return (and create) the output folder of ScienceReady files.

    Args:
        cfg (DotMap): Run configuration.

    Returns:
        Path: ``{output_dir}/ScienceReady``.
    """
    path = Path(cfg.paths.output_dir) / "ScienceReady"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(cfg: DotMap, merger_type: str, sub_id: int, snapshot: int, tile: int, index: int) -> Path:
    """Build the ScienceReady file name; it encodes object, view and background patch.

    Args:
        cfg (DotMap): Run configuration.
        merger_type (str): Merger class label.
        sub_id (int): Subfind ID.
        snapshot (int): Snapshot number.
        tile (int): Background tile ID.
        index (int): Empty-region catalogue row.

    Returns:
        Path: Output path.
    """
    name = f"{merger_type}_sub-{sub_id}_snap-{snapshot}_view-{cfg.render.axes}_BKGPSF_{tile}_{index}.fits"
    return science_ready_dir(cfg) / name


def find_existing(cfg: DotMap, sub_id: int, snapshot: int) -> Path | None:
    """Return an already written ScienceReady file of this object and view, if any.

    Lets an interrupted batch resume without re-rendering finished galaxies.

    Args:
        cfg (DotMap): Run configuration.
        sub_id (int): Subfind ID.
        snapshot (int): Snapshot number.

    Returns:
        Path | None: First matching file, or ``None``.
    """
    pattern = f"*_sub-{sub_id}_snap-{snapshot}_view-{cfg.render.axes}_BKGPSF_*.fits"
    matches = sorted(glob.glob(str(science_ready_dir(cfg) / pattern)))
    return Path(matches[0]) if matches else None


def read_detection_levels(path: Path) -> list[float]:
    """Read the per-band detection levels from a ScienceReady file.

    Args:
        path (Path): ScienceReady FITS file.

    Returns:
        list[float]: ``M_DETLVL`` of VIS, NIR-Y, NIR-J, NIR-H.
    """
    with fits.open(path) as hdul:
        return [float(hdul[band.value].header["M_DETLVL"]) for band in ALL_BANDS]
