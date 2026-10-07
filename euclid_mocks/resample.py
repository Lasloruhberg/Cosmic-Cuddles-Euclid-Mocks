"""Resample images between pixel grids with drizzle and model the dither pattern.

All resampling of the pipeline goes through :func:`drizzle_onto`, which uses
the drizzle 2.x ``resample`` API with a square kernel and ``pixfrac=1``.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from drizzle.resample import Drizzle
from drizzle.utils import calc_pixmap


def make_tan_wcs(
    pixel_scale_arcsec: float,
    crpix: tuple[float, float],
    crval: tuple[float, float] = (0.0, 0.0),
) -> WCS:
    """Build a north-up, east-left gnomonic WCS with square pixels.

    Shared by the source render, the detector frames and the mock tile so all
    grids use the same convention.

    Args:
        pixel_scale_arcsec (float): Pixel side in arcsec.
        crpix (tuple[float, float]): Reference pixel (FITS 1-based, x then y).
        crval (tuple[float, float]): Sky position (RA, Dec) of ``crpix`` in degrees.

    Returns:
        WCS: Celestial WCS.
    """
    wcs = WCS(naxis=2)
    scale_deg = pixel_scale_arcsec / 3600.0
    wcs.wcs.cd = np.array([[-scale_deg, 0.0], [0.0, scale_deg]])
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def drizzle_onto(
    data: np.ndarray, source_wcs: WCS, target_wcs: WCS, out_shape: tuple[int, int]
) -> np.ndarray:
    """Drizzle an image onto another grid while conserving total flux.

    drizzle returns a weighted mean per output pixel; multiplying by the
    accumulated weight (the input-pixel area that landed in the output pixel)
    turns it back into a sum, so values stay in flux per pixel. NaN input is
    treated as zero flux; output pixels nothing landed on come back as NaN.

    Args:
        data (np.ndarray): Input image in flux per pixel.
        source_wcs (WCS): WCS of ``data``.
        target_wcs (WCS): WCS of the output grid.
        out_shape (tuple[int, int]): Output shape (rows, cols).

    Returns:
        np.ndarray: Resampled image in flux per output pixel.
    """
    pixmap = calc_pixmap(source_wcs, target_wcs, shape=data.shape)
    driz = Drizzle(out_shape=out_shape)
    driz.add_image(
        np.nan_to_num(data, nan=0.0), exptime=1, pixmap=pixmap, pixfrac=1.0, weight_map=None
    )
    return np.asarray(driz.out_img, dtype=np.float64) * np.asarray(driz.out_wht, dtype=np.float64)


def pad_hdu(hdu: fits.PrimaryHDU, n_pix: int) -> fits.PrimaryHDU:
    """Return a copy of an image HDU with a zero border and a shifted reference pixel.

    The border keeps drizzle from producing flux artefacts at the edge of the
    source render.

    Args:
        hdu (fits.PrimaryHDU): Image with a celestial WCS.
        n_pix (int): Border width in pixels.

    Returns:
        fits.PrimaryHDU: Padded copy; the sky position of every original pixel is unchanged.
    """
    padded = np.pad(hdu.data, n_pix, mode="constant", constant_values=0.0)
    header = hdu.header.copy()
    header["CRPIX1"] += n_pix
    header["CRPIX2"] += n_pix
    return fits.PrimaryHDU(data=padded, header=header)


def dither_onto_detector(
    flux_hdu: fits.PrimaryHDU,
    detector_pixel_scale_arcsec: float,
    dither_pattern_arcsec: tuple[tuple[float, float], ...],
    planned_shape: int,
    min_overhang: int,
    padding: int,
) -> list[fits.PrimaryHDU]:
    """Resample the true sky onto the detector grid of every dither position.

    This is the noise-free "perfect observation" each exposure starts from.
    Only the sub-pixel part of each dither offset is applied: the integer part
    just moves the source to another detector pixel, which is irrelevant for an
    isolated cutout, while the sub-pixel phase changes the pixel sampling that
    the later stacking has to undo. Every frame has the same square shape:
    ``max(planned_shape, source extent on the detector)`` plus an overhang of
    ``min(max |offset|, min_overhang)`` pixels on each side.

    Args:
        flux_hdu (fits.PrimaryHDU): Source in Jy/pixel with a square-pixel WCS.
        detector_pixel_scale_arcsec (float): Detector pixel scale (VIS 0.1, NISP 0.3).
        dither_pattern_arcsec (tuple): (dx, dy) offset of each dither in arcsec.
        planned_shape (int): Minimum frame side in detector pixels.
        min_overhang (int): Upper limit of the overhang added on each side.
        padding (int): Zero border added to the source before drizzling.

    Returns:
        list[fits.PrimaryHDU]: One detector frame per dither, in Jy/pixel.
    """
    source = pad_hdu(flux_hdu, padding)
    source_wcs = WCS(source.header)
    source_scale_arcsec = proj_plane_pixel_scales(source_wcs)[0] * 3600.0

    offsets_px = np.array(dither_pattern_arcsec) / detector_pixel_scale_arcsec
    overhang = min(np.max(np.abs(offsets_px)), min_overhang)
    source_on_detector = np.ceil(source.data.shape[0] * source_scale_arcsec / detector_pixel_scale_arcsec)
    side = int(np.ceil(max(planned_shape, source_on_detector) + 2 * overhang))

    frames = []
    for offset in offsets_px:
        subpixel = np.mod(offset, 1)
        detector_wcs = make_tan_wcs(
            detector_pixel_scale_arcsec, crpix=(1 + side / 2 + subpixel[0], 1 + side / 2 + subpixel[1])
        )
        data = drizzle_onto(source.data, source_wcs, detector_wcs, (side, side))
        frames.append(fits.PrimaryHDU(data=data, header=detector_wcs.to_header()))
    return frames
