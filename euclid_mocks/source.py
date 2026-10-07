"""Turn a TNG subhalo into a noise-free, flux-calibrated source image.

Covers the steps before any instrument effect: the angular size of the render,
the selection cuts that reject unsuitable subhaloes, and the conversion from
the API's mag/arcsec^2 grid to Jy per pixel on a sky-like WCS.
"""

from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from dotmap import DotMap

from euclid_mocks.constants import JANSKY_AB_ZEROPOINT, TNG_COSMOLOGY, TNG_H
from euclid_mocks.resample import make_tan_wcs


class RejectedGalaxy(ValueError):
    """Signal that a subhalo fails a selection cut and gets no mock image.

    Distinguished from other errors so the batch runner can record the galaxy
    as ``rejected`` rather than ``failed``.
    """


def render_size_arcmin(cfg: DotMap, sub: dict, redshift: float) -> float:
    """Compute the angular side length of the render requested from the API.

    The render spans ``cfg.render.size_in_half_mass_radii`` stellar half-mass
    radii. The comoving radius (ckpc/h) is converted to proper kpc at the
    snapshot redshift and then to arcmin with the TNG cosmology. The value is
    rounded to 1e-3 arcmin because the API request and the later flux
    conversion must use exactly the same number.

    Args:
        cfg (DotMap): Run configuration.
        sub (dict): Subhalo record from the API.
        redshift (float): Snapshot redshift.

    Returns:
        float: Render side length in arcmin, rounded to 3 decimals.

    Raises:
        ValueError: If the redshift is too small for a meaningful angular scale.
    """
    if redshift < cfg.selection.min_redshift:
        raise ValueError(f"z={redshift} too low to compute an angular size")
    kpc_per_arcsec = TNG_COSMOLOGY.kpc_proper_per_arcmin(redshift).value / 60.0
    comoving_kpc = cfg.render.size_in_half_mass_radii * sub["halfmassrad_stars"] / TNG_H
    proper_kpc = comoving_kpc / (1.0 + redshift)
    return round(proper_kpc / kpc_per_arcsec / 60.0, 3)


def check_subhalo_selection(cfg: DotMap, sub: dict, size_arcmin: float) -> None:
    """Reject subhaloes that are too poorly resolved or too small to image.

    Applied before any render is requested, so rejected objects cost no
    render-server time.

    Args:
        cfg (DotMap): Run configuration.
        sub (dict): Subhalo record from the API.
        size_arcmin (float): Render side length from :func:`render_size_arcmin`.

    Raises:
        RejectedGalaxy: If the particle count, stellar mass or size is below the limit.
    """
    sel = cfg.selection
    if sub["len_stars"] < sel.min_star_particles:
        raise RejectedGalaxy(f"too few star particles: {sub['len_stars']}")
    if sub["mass_stars"] < sel.min_stellar_mass_msun * TNG_H / 1e10:
        raise RejectedGalaxy(f"stellar mass too low: {sub['mass_stars'] * 1e10 / TNG_H:.3e} Msun")
    if size_arcmin < sel.min_size_arcmin:
        raise RejectedGalaxy(f"render too small: {size_arcmin} arcmin")


def load_render(path: Path) -> np.ndarray:
    """Read the surface-brightness grid of a cached ``vis.hdf5`` render.

    Args:
        path (Path): HDF5 file from :func:`euclid_mocks.tng_api.download_render`.

    Returns:
        np.ndarray: Square array in mag/arcsec^2.
    """
    with h5py.File(path, mode="r") as f:
        return f["grid"][()]


def check_render_brightness(cfg: DotMap, surface_brightness: np.ndarray) -> None:
    """Reject renders whose brightest pixel is fainter than the configured limit.

    Such objects would vanish in the background and only add noise to the
    training set.

    Args:
        cfg (DotMap): Run configuration.
        surface_brightness (np.ndarray): Render in mag/arcsec^2.

    Raises:
        RejectedGalaxy: If ``nanmin(surface_brightness)`` exceeds the limit.
    """
    peak = np.nanmin(surface_brightness)
    if peak > cfg.selection.faintest_allowed_peak_sb:
        raise RejectedGalaxy(f"too faint: brightest pixel {peak:.2f} mag/arcsec^2")


def surface_brightness_to_flux_hdu(surface_brightness: np.ndarray, size_arcmin: float) -> fits.PrimaryHDU:
    """Convert a mag/arcsec^2 render into Jy per pixel on a tangent-plane WCS.

    This is the noise-free "true sky" that the detector stages dither and
    observe. The WCS is centred on (RA, Dec) = (0, 0) with north up and east
    left; the reference pixel is the image centre in FITS 1-based indexing.

    Args:
        surface_brightness (np.ndarray): Square render in AB mag/arcsec^2.
        size_arcmin (float): Side length the render was requested with.

    Returns:
        fits.PrimaryHDU: Flux in Jy/pixel with its WCS.
    """
    n_pixels = surface_brightness.shape[0]
    if surface_brightness.shape != (n_pixels, n_pixels):
        raise ValueError(f"render is not square: {surface_brightness.shape}")
    pixel_scale_arcsec = 60.0 * size_arcmin / n_pixels
    flux = pixel_scale_arcsec**2 * JANSKY_AB_ZEROPOINT * 10 ** (surface_brightness / -2.5)
    wcs = make_tan_wcs(pixel_scale_arcsec, crpix=(n_pixels / 2 + 1, n_pixels / 2 + 1))
    return fits.PrimaryHDU(data=flux, header=wcs.to_header())
