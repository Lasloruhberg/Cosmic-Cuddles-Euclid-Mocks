"""Model single Euclid exposures and combine them into a stack.

VIS is modelled as a CCD integration (photon noise only, read noise is
negligible at these exposure times). NISP is modelled with the MACC
up-the-ramp readout and the on-board group-difference slope estimator,
which is what sets its noise properties.
"""

import warnings

import numpy as np
from loguru import logger

from euclid_mocks.constants import JANSKY_AB_ZEROPOINT, NISP_POISSON_LAMBDA_CAP


def flux_to_electrons(
    flux_jy: np.ndarray, exp_time: float, gain: float, zeropoint: float, adu_saturation: float
) -> np.ndarray:
    """Convert a Jy/pixel image into collected electrons for one VIS exposure.

    Uses the VIS photometric zeropoint, defined in ADU/s:
    ``ADU = t * 10**(ZP/2.5) * F / 3631 Jy``, then ``e- = ADU * gain``.
    Saturation is only reported, not applied.

    Args:
        flux_jy (np.ndarray): Image in Jy/pixel (NaN outside the source footprint).
        exp_time (float): Exposure time in s.
        gain (float): Gain in e-/ADU.
        zeropoint (float): AB zeropoint for ADU/s.
        adu_saturation (float): ADU level above which a pixel saturates.

    Returns:
        np.ndarray: Expected electrons per pixel.
    """
    adu = (exp_time * 10 ** (zeropoint / 2.5) * flux_jy / JANSKY_AB_ZEROPOINT).astype(np.float64)
    saturated = np.nan_to_num(adu, nan=0.0) > adu_saturation
    if saturated.any():
        fraction = 100 * saturated.sum() / np.sum(~np.isnan(adu))
        logger.warning(f"{fraction:.3f}% of pixels exceed the ADU saturation level (not clipped)")
    return adu * gain


def electrons_to_adu_per_s(electrons: np.ndarray, exp_time: float, gain: float) -> np.ndarray:
    """Convert collected electrons back to a count rate in ADU/s.

    This is the unit of the VIS stack and the one its zeropoint refers to.

    Args:
        electrons (np.ndarray): Electrons per pixel.
        exp_time (float): Exposure time in s.
        gain (float): Gain in e-/ADU.

    Returns:
        np.ndarray: Count rate in ADU/s.
    """
    return (electrons / gain).astype(np.float64) / exp_time


def add_poisson_noise(electrons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a Poisson realisation of an expected-electron image, keeping NaNs.

    Args:
        electrons (np.ndarray): Expected electrons per pixel.
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        np.ndarray: Noisy electrons as float64; NaN where the input was NaN.
    """
    nan_mask = np.isnan(electrons)
    noisy = rng.poisson(np.where(nan_mask, 0.0, electrons)).astype(np.float64)
    noisy[nan_mask] = np.nan
    return noisy


def observe_vis_exposure(
    flux_jy: np.ndarray,
    exp_time: float,
    gain: float,
    zeropoint: float,
    adu_saturation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one VIS exposure: Jy -> electrons -> Poisson -> ADU/s.

    Args:
        flux_jy (np.ndarray): Detector frame in Jy/pixel.
        exp_time (float): Exposure time in s.
        gain (float): Gain in e-/ADU.
        zeropoint (float): AB zeropoint for ADU/s.
        adu_saturation (float): ADU saturation level (reported only).
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        np.ndarray: Noisy exposure in ADU/s.
    """
    electrons = flux_to_electrons(flux_jy, exp_time, gain, zeropoint, adu_saturation)
    return electrons_to_adu_per_s(add_poisson_noise(electrons, rng), exp_time, gain)


def flux_to_electron_rate_nisp(flux_jy: np.ndarray, inverse_sensitivity: float) -> np.ndarray:
    """Convert a Jy/pixel image to a NISP electron rate with the Q1 inverse sensitivity.

    ``F[e-/s] = F[micro-Jy] / IS``; the IS already includes the quantum efficiency.

    Args:
        flux_jy (np.ndarray): Detector frame in Jy/pixel.
        inverse_sensitivity (float): Band IS in micro-Jy per e-/s.

    Returns:
        np.ndarray: Expected electrons per second.
    """
    return flux_jy * 1e6 / inverse_sensitivity


def _poisson_lambda(expected: np.ndarray) -> np.ndarray:
    lam = np.nan_to_num(np.asarray(expected, dtype=np.float64), nan=0.0, posinf=NISP_POISSON_LAMBDA_CAP, neginf=0.0)
    return np.clip(lam, 0.0, NISP_POISSON_LAMBDA_CAP)


def observe_nisp_macc(
    electron_rate: np.ndarray,
    n_groups: int,
    n_frames: int,
    n_dropped: int,
    frame_time: float,
    read_noise_e: float,
    gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a NISP MACC(n_g, n_f, n_d) ramp and apply the on-board slope estimator.

    The ramp is built frame by frame: each frame adds a Poisson draw of the
    accumulated signal plus Gaussian read noise (both converted to ADU);
    dropped frames add signal but are not read. Frames are averaged per group,
    and the signal is estimated from the group differences with the NISP
    maximum-likelihood group-difference estimator ("Eq. 11" of the original
    implementation). The estimate is rescaled from one group interval to the
    full integration and converted from ADU back to electrons; ground
    calibrations (bias, non-linearity, dark, flats) are irrelevant here.

    Args:
        electron_rate (np.ndarray): Expected e-/s per pixel (NaN treated as 0).
        n_groups (int): Number of groups.
        n_frames (int): Frames averaged per group.
        n_dropped (int): Frames dropped between groups.
        frame_time (float): Frame time in s.
        read_noise_e (float): Read noise per frame in e-.
        gain (float): Gain in ADU/e-.
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        np.ndarray: Estimated signal per pixel in electrons.
    """
    shape = electron_rate.shape
    alpha = 1 / n_frames
    group_time = (n_frames + n_dropped) * frame_time
    beta = (2 * read_noise_e**2 * group_time) / (n_frames * (1 + alpha))

    group_means = np.zeros((n_groups, *shape))
    accumulated_adu = np.zeros(shape)
    for g in range(n_groups):
        frame_sum = np.zeros(shape)
        for _ in range(n_frames):
            photons = rng.poisson(_poisson_lambda(electron_rate * frame_time)).astype(np.float32)
            read_noise = rng.normal(0, read_noise_e, size=shape)
            frame_sum += accumulated_adu + photons * gain + read_noise * gain
            accumulated_adu += photons * gain
        group_means[g] = frame_sum / n_frames
        dropped = rng.poisson(_poisson_lambda(electron_rate * frame_time * n_dropped)).astype(np.float32)
        accumulated_adu += dropped * gain

    delta = group_means[1:] - group_means[:-1]
    sum_term = np.sum((delta + beta) ** 2, axis=0)
    inner = 1 + (4 * group_time**2 * sum_term) / ((n_groups - 1) * (1 + alpha) ** 2)
    slope = ((1 + alpha) / (2 * group_time)) * (np.sqrt(inner) - 1) - beta
    # The estimator yields the signal per group interval; scale it to the full integration.
    return slope * (n_groups - 1) / gain


def median_stack(frames: list[np.ndarray]) -> np.ndarray:
    """Median-combine aligned exposures, as MER does, and zero uncovered pixels.

    Args:
        frames (list[np.ndarray]): Exposures on a common grid.

    Returns:
        np.ndarray: Per-pixel median; 0 where no frame has data.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        combined = np.nanmedian(np.stack(frames), axis=0)
    return np.nan_to_num(combined, nan=0.0)


def crop_to_footprint(data: np.ndarray, header, brim_trim: int):
    """Crop a stack to the square bounding box of its non-zero pixels.

    Removes the empty detector margin around the source. ``brim_trim``
    pixels are shaved off each side of the bounding box to drop the
    partially covered edge, and the shorter side is then extended to make
    the cutout square.

    Args:
        data (np.ndarray): Stacked image.
        header (fits.Header): Header carrying the WCS of ``data``.
        brim_trim (int): Pixels removed inside each edge of the bounding box.

    Returns:
        tuple[np.ndarray, fits.Header]: Cropped data and a header with shifted CRPIX.
    """
    mask = np.isfinite(data) & (data != 0)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    rmin, rmax = max(0, rows[0] + brim_trim), min(data.shape[0] - 1, rows[-1] - brim_trim)
    cmin, cmax = max(0, cols[0] + brim_trim), min(data.shape[1] - 1, cols[-1] - brim_trim)
    if rmax - rmin > cmax - cmin:
        cmax = cmin + (rmax - rmin)
    else:
        rmax = rmin + (cmax - cmin)
    cropped = data[rmin : rmax + 1, cmin : cmax + 1]
    new_header = header.copy()
    new_header["CRPIX1"] -= cmin
    new_header["CRPIX2"] -= rmin
    return cropped, new_header
