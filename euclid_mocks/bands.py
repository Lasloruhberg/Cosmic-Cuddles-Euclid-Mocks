"""Produce the stacked, noisy mock image of one galaxy in each Euclid band.

VIS is built first and defines the reference grid; every NISP band is
drizzled straight onto that grid so all bands share shape and WCS.
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from dotmap import DotMap

from euclid_mocks.constants import (
    NISP_FRAME_TIME_S,
    NISP_GAIN_ADU_PER_E,
    NISP_INVERSE_SENSITIVITY,
    NISP_N_DROPPED,
    NISP_N_FRAMES,
    NISP_N_GROUPS,
    NISP_PIXEL_SCALE_ARCSEC,
    NISP_READ_NOISE_E,
    NISP_ZEROPOINT,
    Q1_DITHER_PATTERN_ARCSEC,
    VIS_ADU_SATURATION,
    VIS_GAIN_E_PER_ADU,
    VIS_N_SHORT_EXPOSURE_DITHERS,
    VIS_NOMINAL_EXPOSURE_S,
    VIS_PIXEL_SCALE_ARCSEC,
    VIS_SHORT_EXPOSURE_S,
    VIS_ZEROPOINT_ADU_PER_S,
    EuclidBand,
)
from euclid_mocks.detector import (
    crop_to_footprint,
    flux_to_electron_rate_nisp,
    median_stack,
    observe_nisp_macc,
    observe_vis_exposure,
)
from euclid_mocks.resample import dither_onto_detector, drizzle_onto


def _dither(cfg: DotMap, flux_hdu: fits.PrimaryHDU, pixel_scale: float) -> list[fits.PrimaryHDU]:
    d = cfg.detector
    return dither_onto_detector(
        flux_hdu,
        pixel_scale,
        Q1_DITHER_PATTERN_ARCSEC,
        planned_shape=d.planned_shape_px,
        min_overhang=d.min_overhang_px,
        padding=d.source_padding_px,
    )


def observe_vis(cfg: DotMap, flux_hdu: fits.PrimaryHDU, rng: np.random.Generator) -> fits.PrimaryHDU:
    """Simulate the stacked VIS image of one galaxy and define the reference grid.

    The true sky is dithered onto the 0.1"/px detector. Each dither gets a
    nominal 566 s exposure, and the first two also get a short 95 s exposure;
    every exposure is observed with photon noise, drizzled onto the dither-0
    grid and the six frames are median-combined. The stack is cropped to the
    source footprint (minus a brim) and finally cut to a square centred on the
    galaxy position (0, 0), whose WCS all other bands are resampled onto.

    Args:
        cfg (DotMap): Run configuration.
        flux_hdu (fits.PrimaryHDU): True sky in Jy/pixel (VIS filter).
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        fits.PrimaryHDU: Stack in ADU/s with WCS and ``MAGZERO``.
    """
    dithers = _dither(cfg, flux_hdu, VIS_PIXEL_SCALE_ARCSEC)
    reference_wcs = WCS(dithers[0].header)
    reference_shape = dithers[0].data.shape

    frames = []
    for i, frame in enumerate(dithers):
        exposures = [VIS_NOMINAL_EXPOSURE_S]
        if i < VIS_N_SHORT_EXPOSURE_DITHERS:
            exposures.append(VIS_SHORT_EXPOSURE_S)
        for exp_time in exposures:
            observed = observe_vis_exposure(
                frame.data, exp_time, VIS_GAIN_E_PER_ADU, VIS_ZEROPOINT_ADU_PER_S, VIS_ADU_SATURATION, rng
            )
            frames.append(drizzle_onto(observed, WCS(frame.header), reference_wcs, reference_shape))

    cropped, header = crop_to_footprint(
        median_stack(frames), reference_wcs.to_header(), cfg.detector.vis_brim_trim_px
    )
    ny, nx = cropped.shape
    if nx != ny:
        raise ValueError(f"VIS footprint crop is not square: {cropped.shape}")
    # Shaving one more pixel per side removes the partially covered edge left by the crop.
    cutout = Cutout2D(
        cropped,
        position=SkyCoord(0 * u.deg, 0 * u.deg),
        size=(ny - 2, nx - 2),
        wcs=WCS(header),
        mode="partial",
        fill_value=0,
    )
    header.update(cutout.wcs.to_header())
    header["MAGZERO"] = VIS_ZEROPOINT_ADU_PER_S
    return fits.PrimaryHDU(data=cutout.data, header=header)


def observe_nisp(
    cfg: DotMap,
    flux_hdu: fits.PrimaryHDU,
    band: EuclidBand,
    vis_hdu: fits.PrimaryHDU,
    rng: np.random.Generator,
) -> fits.PrimaryHDU:
    """Simulate the stacked image of one galaxy in a NISP band on the VIS grid.

    The true sky is dithered onto the 0.3"/px NISP detector; each dither is
    converted to an electron rate, observed with the MACC ramp model, drizzled
    onto the VIS reference grid and the four frames are median-combined.

    Args:
        cfg (DotMap): Run configuration.
        flux_hdu (fits.PrimaryHDU): True sky in Jy/pixel for this band.
        band (EuclidBand): NISP band.
        vis_hdu (fits.PrimaryHDU): Output of :func:`observe_vis`, defines the grid.
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        fits.PrimaryHDU: Stack in e-/s-like MER units with VIS WCS and ``MAGZERO``.
    """
    target_wcs = WCS(vis_hdu.header)
    target_shape = vis_hdu.data.shape
    frames = []
    for frame in _dither(cfg, flux_hdu, NISP_PIXEL_SCALE_ARCSEC):
        rate = flux_to_electron_rate_nisp(frame.data, NISP_INVERSE_SENSITIVITY[band])
        observed = observe_nisp_macc(
            rate,
            n_groups=NISP_N_GROUPS,
            n_frames=NISP_N_FRAMES,
            n_dropped=NISP_N_DROPPED,
            frame_time=NISP_FRAME_TIME_S,
            read_noise_e=NISP_READ_NOISE_E,
            gain=NISP_GAIN_ADU_PER_E,
            rng=rng,
        )
        frames.append(drizzle_onto(observed, WCS(frame.header), target_wcs, target_shape))

    header = target_wcs.to_header()
    header["MAGZERO"] = NISP_ZEROPOINT[band]
    return fits.PrimaryHDU(data=median_stack(frames), header=header)
