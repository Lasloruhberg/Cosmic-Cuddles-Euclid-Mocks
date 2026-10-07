"""Place a mock galaxy into real (or mock) Euclid MER background and PSF.

A background position is chosen once per galaxy from the empty-region
catalogue and reused for every band, so the four bands see the same patch of
sky. Each band is then convolved with the matching kernel from the
simulation's delta-function PSF to the local Euclid PSF, rescaled to the
mosaic zeropoint and added to the background cutout.
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.convolution import convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger
from photutils.background import StdBackgroundRMS
from photutils.psf_matching import SplitCosineBellWindow, make_kernel

from euclid_mocks.constants import EuclidBand


@dataclass(frozen=True)
class BackgroundPosition:
    """Identify where in which MER tile a mock galaxy is placed.

    Attributes:
        tile (int): MER tile ID.
        ra (float): RA of the cutout centre in degrees (after the random offset).
        dec (float): Dec of the cutout centre in degrees.
        index (int): Row index of the chosen entry in the tile's empty-region CSV.
    """

    tile: int
    ra: float
    dec: float
    index: int

    @property
    def coord(self) -> SkyCoord:
        """Return the cutout centre as a SkyCoord."""
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)


def find_mer_file(cfg: DotMap, tile: int, band: EuclidBand, product: str) -> Path:
    """Locate a MER product of a tile following the Q1 folder layout.

    Args:
        cfg (DotMap): Run configuration.
        tile (int): MER tile ID.
        band (EuclidBand): Band of the product.
        product (str): ``"BGSUB-MOSAIC"`` or ``"GRID-PSF"``.

    Returns:
        Path: The (alphabetically first) matching FITS file.

    Raises:
        FileNotFoundError: If no file matches.
    """
    pattern = (
        f"{cfg.paths.euclid_mer_dir}/{tile}/{band.instrument_folder}/"
        f"EUC_MER_{product}-{band.value}_TILE{tile}*.fits"
    )
    return first_match(pattern)


def first_match(pattern: str) -> Path:
    """Resolve a MER glob pattern to one file, warning when the choice is ambiguous.

    Q1 tile folders hold one product per band, but DR1 deep-field folders can
    hold several (e.g. different stacks or processing versions). The
    alphabetically first match is used then and every candidate is logged, so
    an unintended choice is visible.

    Args:
        pattern (str): Glob pattern.

    Returns:
        Path: The alphabetically first match.

    Raises:
        FileNotFoundError: If nothing matches.
    """
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no MER file matches {pattern}")
    if len(matches) > 1:
        logger.warning(f"{len(matches)} files match {pattern}; using {matches[0]}. Candidates: {matches}")
    return Path(matches[0])


def psf_stamp(cfg: DotMap, tile: int, band: EuclidBand, position: SkyCoord) -> np.ndarray:
    """Return the MER PSF stamp closest to a sky position.

    The ``GRID-PSF`` product stores a grid of stamps in extension 1 (stamp size
    in ``STMPSIZE``) and the 1-based stamp centres in the ``x``/``y`` columns
    of extension 2.

    Args:
        cfg (DotMap): Run configuration.
        tile (int): MER tile ID.
        band (EuclidBand): Band.
        position (SkyCoord): Position of the mock galaxy.

    Returns:
        np.ndarray: Square PSF stamp with an odd side length.
    """
    with fits.open(find_mer_file(cfg, tile, band, "GRID-PSF")) as hdul:
        grid, header, table = hdul[1].data, hdul[1].header, hdul[2].data
        stamp_size = int(header["STMPSIZE"])
        x_pos, y_pos = position.to_pixel(WCS(header), origin=1)
        nearest = np.argmin(np.hypot(table["x"] - x_pos, table["y"] - y_pos))
        x, y = table["x"][nearest], table["y"][nearest]
        stamp = Cutout2D(grid, position=(x - 1, y - 1), size=stamp_size).data.copy()
    centre = (stamp_size - 1) // 2
    if not np.isclose(stamp[centre, centre], np.nanmax(stamp)):
        logger.warning(f"{band.value} PSF stamp peak is not at its centre pixel")
    return stamp


def psf_matching_kernel(cfg: DotMap, tile: int, band: EuclidBand, position: SkyCoord) -> np.ndarray:
    """Build the kernel that turns the simulation PSF into the local Euclid PSF.

    The TNG render has no PSF, i.e. a delta function, so the kernel is the
    regularised ratio of the Euclid stamp to a centred delta, tapered with a
    split cosine bell window whose shape is set per band in ``cfg.psf.window``.

    Args:
        cfg (DotMap): Run configuration.
        tile (int): MER tile ID.
        band (EuclidBand): Band.
        position (SkyCoord): Position of the mock galaxy.

    Returns:
        np.ndarray: Matching kernel with the stamp's shape.
    """
    stamp = psf_stamp(cfg, tile, band, position)
    if stamp.shape[0] != stamp.shape[1]:
        raise ValueError(f"non-square PSF stamp {stamp.shape}")
    delta = np.zeros_like(stamp)
    delta[(stamp.shape[0] - 1) // 2, (stamp.shape[1] - 1) // 2] = 1.0
    window_cfg = cfg.psf.window[band.value]
    window = SplitCosineBellWindow(alpha=window_cfg.alpha, beta=window_cfg.beta)
    return make_kernel(delta, stamp, window=window)


def _read_empty_regions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Catalogues are written with the pandas index as an unnamed first column.
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "index"})
    return df


def _draw_entry(cfg: DotMap, min_half_size: float, min_distance: float, excluded: set, rng):
    """Pick an empty-region catalogue row that is far enough from borders and objects."""
    files = sorted(glob.glob(f"{cfg.paths.empty_region_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"no empty-region CSVs in {cfg.paths.empty_region_dir}")
    fallback = None
    n_fallback_tiles = 0
    for i in rng.permutation(len(files)):
        tile_name = Path(files[i]).name.split("_")[1]
        if tile_name in excluded:
            continue
        df = _read_empty_regions(files[i])
        far_from_border = df[df["border_distance"] > min_half_size]
        if far_from_border.empty:
            continue
        clean = far_from_border[far_from_border["min_object_distance"] > min_distance]
        if not clean.empty:
            return clean.sample(n=1, random_state=rng).iloc[0]
        fallback = far_from_border.loc[far_from_border["min_object_distance"].idxmax()]
        n_fallback_tiles += 1
        # Scanning every tile for a perfectly clean patch is slow; settle for the least crowded one.
        if n_fallback_tiles >= cfg.background.max_fallback_tiles:
            break
    if fallback is None:
        raise RuntimeError("no empty region is far enough from the tile border for this image size")
    logger.info("no clean empty region found; using the one with the largest object distance")
    return fallback


def _offset_position(cfg: DotMap, entry, min_half_size: float, min_distance: float, rng) -> BackgroundPosition:
    """Shift the catalogue position randomly so repeated draws do not reuse the same pixels."""
    tile = int(entry["Folder_name"])
    border_limit = entry["border_distance"] - min_half_size
    if border_limit <= 0:
        raise ValueError(f"empty region {entry['index']} of tile {tile} is too close to the border")
    max_offset = max(
        cfg.background.min_offset_factor * entry["min_object_distance"],
        entry["min_object_distance"] - min_distance,
    )
    radius = rng.uniform(0.0, min(border_limit, max_offset))
    angle = rng.uniform(0, 2 * np.pi)
    mosaic_wcs = WCS(fits.getheader(find_mer_file(cfg, tile, EuclidBand.VIS, "BGSUB-MOSAIC")))
    x, y = mosaic_wcs.world_to_pixel(SkyCoord(entry["pos_ra"] * u.deg, entry["pos_dec"] * u.deg))
    sky = mosaic_wcs.pixel_to_world(x + radius * np.cos(angle), y + radius * np.sin(angle))
    return BackgroundPosition(tile=tile, ra=float(sky.ra.deg), dec=float(sky.dec.deg), index=int(entry["index"]))


def background_cutout(
    cfg: DotMap, position: BackgroundPosition, band: EuclidBand, shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """Cut the background-subtracted MER mosaic around a position.

    ``mode="strict"`` guarantees the cutout has exactly ``shape`` (or raises),
    which keeps every band on the VIS reference geometry.

    Args:
        cfg (DotMap): Run configuration.
        position (BackgroundPosition): Cutout centre.
        band (EuclidBand): Band of the mosaic.
        shape (tuple[int, int]): Cutout shape (rows, cols).

    Returns:
        tuple[np.ndarray, float]: Cutout data and the mosaic ``MAGZERO``.
    """
    with fits.open(find_mer_file(cfg, position.tile, band, "BGSUB-MOSAIC")) as hdul:
        cutout = Cutout2D(
            hdul[0].data, position=position.coord, size=shape, wcs=WCS(hdul[0].header), mode="strict"
        )
        return cutout.data.copy(), float(hdul[0].header["MAGZERO"])


def _is_clean(cfg: DotMap, position: BackgroundPosition, shape: tuple[int, int]) -> bool:
    max_bad = cfg.background.max_bad_pixel_fraction
    vis, _ = background_cutout(cfg, position, EuclidBand.VIS, shape)
    check, _ = background_cutout(cfg, position, EuclidBand(cfg.background.quality_check_band), shape)
    vis_bad = vis == 0
    return bool(
        np.any(~vis_bad) and vis_bad.mean() <= max_bad and (check == 0).mean() <= max_bad
    )


def choose_background(cfg: DotMap, shape: tuple[int, int], rng: np.random.Generator) -> BackgroundPosition:
    """Choose the sky patch a mock galaxy of a given size is placed into.

    Draws an empty region whose distance to the tile border exceeds
    ``border_size_factor * min(shape)`` and, preferably, whose distance to
    the nearest object exceeds ``object_distance_factor`` times that. The
    position is shifted by a random offset. Patches with too many zero
    (uncovered) pixels in VIS or the quality-check band are redrawn from other
    tiles; after ``max_retries`` redraws the last patch is used anyway.

    Args:
        cfg (DotMap): Run configuration.
        shape (tuple[int, int]): Shape of the VIS mock image.
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        BackgroundPosition: Tile and sky position for all bands.
    """
    min_half_size = cfg.background.border_size_factor * min(shape)
    min_distance = cfg.background.object_distance_factor * min_half_size
    excluded: set[str] = set()
    for attempt in range(cfg.background.max_retries + 1):
        entry = _draw_entry(cfg, min_half_size, min_distance, excluded, rng)
        position = _offset_position(cfg, entry, min_half_size, min_distance, rng)
        if _is_clean(cfg, position, shape):
            return position
        excluded.add(str(position.tile))
        logger.info(f"background of tile {position.tile} has too many empty pixels, redrawing")
    logger.warning(f"no clean background after {cfg.background.max_retries} redraws; using the last one")
    return position


def inject_background(
    cfg: DotMap,
    image: np.ndarray,
    header: fits.Header,
    band: EuclidBand,
    position: BackgroundPosition,
) -> tuple[np.ndarray, fits.Header, float, float]:
    """Convolve a band image with the Euclid PSF and add it to the MER background.

    The convolved image is rescaled to the mosaic zeropoint so that source and
    background share units. Background pixels that are 0 (no coverage) stay 0.
    The detection level is the mean of the central 8x8 pixels of the convolved
    source divided by the sigma-clipped standard deviation of the background.

    Args:
        cfg (DotMap): Run configuration.
        image (np.ndarray): Noisy band stack (pre-PSF), on the VIS grid.
        header (fits.Header): Its header; must hold ``MAGZERO``. Updated in place.
        band (EuclidBand): Band.
        position (BackgroundPosition): Background patch from :func:`choose_background`.

    Returns:
        tuple: ``(combined image, header, detection level, background noise)``.
    """
    bkg = cfg.background
    background, bkg_magzero = background_cutout(cfg, position, band, image.shape)
    kernel = psf_matching_kernel(cfg, position.tile, band, position.coord)
    convolved = convolve_fft(np.nan_to_num(image, nan=0.0), kernel, normalize_kernel=True, allow_huge=True)

    if not np.isclose(bkg_magzero, header["MAGZERO"]):
        convolved *= 10 ** (-(header["MAGZERO"] - bkg_magzero) / 2.5)
        header["MAGZERO"] = (bkg_magzero, "rescaled to the background zeropoint")

    cy, cx = convolved.shape[0] // 2, convolved.shape[1] // 2
    h = bkg.detection_box_half_size_px
    signal = np.mean(convolved[cy - h : cy + h, cx - h : cx + h])
    rms = StdBackgroundRMS(sigma_clip=SigmaClip(sigma=bkg.noise_sigma_clip, maxiters=bkg.noise_sigma_clip_maxiters))
    noise = float(rms(np.nan_to_num(background)))
    detection_level = signal / noise if noise > 0 else np.nan

    combined = convolved + background
    combined[background == 0] = 0

    header["BKG_TILE"] = (position.tile, "Background Tile that was cutout")
    header["BKG_RA"] = (position.ra, "Background RA at cutout center [deg]")
    header["BKG_DEC"] = (position.dec, "Background DEC at cutout center [deg]")
    header["BKG_NOIS"] = (noise, "determined bkg noise level")
    header["M_DETLVL"] = (detection_level, "average 8x8 center signal vs bkg noise")
    return combined, header, float(detection_level), noise
