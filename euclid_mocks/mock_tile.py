"""Generate a small synthetic MER tile and a synthetic galaxy for offline runs.

The tile mimics the Q1 MER product layout (background-subtracted mosaics,
PSF grids, segmentation map and empty-region catalogue) so the background
stage runs unchanged without access to Euclid data. Depths and PSF widths
are approximate Q1 values from ``cfg.mock_tile``; nothing here is real data.
"""

from pathlib import Path

import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.modeling.models import Moffat2D, Sersic2D
from astropy.table import Table
from astropy.wcs import WCS
from dotmap import DotMap
from loguru import logger
from scipy.ndimage import label as ndi_label

from euclid_mocks.constants import ALL_BANDS, JANSKY_AB_ZEROPOINT, EuclidBand
from euclid_mocks.empty_regions import find_empty_regions, write_empty_region_catalog
from euclid_mocks.resample import make_tan_wcs


def moffat_psf(fwhm_px: float, beta: float, size: int) -> np.ndarray:
    """Return a unit-sum Moffat PSF stamp centred on the middle pixel.

    Args:
        fwhm_px (float): FWHM in pixels.
        beta (float): Moffat power index.
        size (int): Odd stamp side length.

    Returns:
        np.ndarray: PSF stamp.
    """
    gamma = fwhm_px / (2 * np.sqrt(2 ** (1 / beta) - 1))
    centre = (size - 1) / 2
    y, x = np.mgrid[:size, :size]
    psf = Moffat2D(1.0, centre, centre, gamma, beta)(x, y)
    return psf / psf.sum()


def _pixel_noise(band_cfg: DotMap, pixel_scale: float) -> float:
    """Per-pixel noise implied by an n-sigma point-source depth in a circular aperture."""
    aperture_px = np.pi * (band_cfg.depth_aperture_arcsec / 2 / pixel_scale) ** 2
    aperture_flux = 10 ** (-0.4 * (band_cfg.depth_mag - band_cfg.magzero))
    return aperture_flux / band_cfg.depth_nsigma / np.sqrt(aperture_px)


def _source_shapes(mt: DotMap, rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Draw unit-flux galaxy and star images plus their AB magnitudes."""
    size = mt.size_px
    y, x = np.mgrid[:size, :size]
    shapes = []
    for _ in range(mt.n_galaxies):
        model = Sersic2D(
            amplitude=1.0,
            r_eff=rng.uniform(2.0, 8.0),
            n=rng.uniform(1.0, 4.0),
            x_0=rng.uniform(20, size - 20),
            y_0=rng.uniform(20, size - 20),
            ellip=rng.uniform(0.0, 0.6),
            theta=rng.uniform(0, np.pi),
        )
        image = model(x, y)
        shapes.append(image / image.sum())
    for _ in range(mt.n_stars):
        image = np.zeros((size, size))
        image[rng.integers(20, size - 20), rng.integers(20, size - 20)] = 1.0
        shapes.append(image)
    mags = rng.uniform(*mt.mag_range, size=len(shapes))
    return shapes, mags


def _write_psf_grid(path: Path, psf: np.ndarray, tile_wcs: WCS, mt: DotMap) -> None:
    """Write a GRID-PSF file: stamp grid in ext 1 (with STMPSIZE and a sky WCS), centres in ext 2."""
    stamp = psf.shape[0]
    n = mt.psf_grid_cells
    grid = np.tile(psf, (n, n))
    # The grid WCS maps each stamp centre to the sky position of its cell on the tile.
    tile_px_per_grid_px = (mt.size_px / n) / stamp
    grid_wcs = make_tan_wcs(
        mt.pixel_scale_arcsec * tile_px_per_grid_px,
        crpix=((n * stamp + 1) / 2, (n * stamp + 1) / 2),
        crval=tuple(tile_wcs.wcs.crval),
    )
    header = grid_wcs.to_header()
    header["STMPSIZE"] = stamp
    centres = np.arange(n) * stamp + (stamp + 1) / 2
    cx, cy = np.meshgrid(centres, centres)
    ra, dec = grid_wcs.all_pix2world(cx.ravel(), cy.ravel(), 1)
    table = Table({"x": cx.ravel(), "y": cy.ravel(), "RA": ra, "DEC": dec})
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(grid, header=header), fits.BinTableHDU(table)]
    ).writeto(path, overwrite=True)


def build_mock_tile(cfg: DotMap, out_dir: Path | None = None) -> Path:
    """Write a complete synthetic MER tile with its empty-region catalogue.

    Draws Sérsic galaxies and stars, convolves them with a per-band Moffat
    PSF, adds Gaussian noise at the configured depth and writes, in the Q1
    folder layout under ``out_dir``: ``MER/{tile}/VIS|NISP`` mosaics and PSF
    grids, a segmentation map, and ``EmptyRegionsCatalog/TILE_{tile}_empty_regions.csv``
    built with :func:`euclid_mocks.empty_regions.find_empty_regions`.

    Args:
        cfg (DotMap): Run configuration (uses ``cfg.mock_tile``).
        out_dir (Path | None): Target folder; defaults to ``cfg.paths.mock_tile_dir``.

    Returns:
        Path: The folder that was written.
    """
    mt = cfg.mock_tile
    out_dir = Path(out_dir if out_dir is not None else cfg.paths.mock_tile_dir)
    tile = mt.tile_id
    rng = np.random.default_rng(mt.seed)
    centre = (mt.size_px + 1) / 2
    tile_wcs = make_tan_wcs(mt.pixel_scale_arcsec, crpix=(centre, centre), crval=(mt.ra_deg, mt.dec_deg))
    shapes, mags = _source_shapes(mt, rng)

    detected = np.zeros((mt.size_px, mt.size_px), dtype=bool)
    for band in ALL_BANDS:
        band_cfg = mt.bands[band.value]
        psf = moffat_psf(band_cfg.psf_fwhm_arcsec / mt.pixel_scale_arcsec, mt.psf_moffat_beta, band_cfg.psf_stamp_px)
        fluxes = 10 ** (-0.4 * (mags - band_cfg.magzero))
        model = convolve_fft(sum(f * s for f, s in zip(fluxes, shapes)), psf, normalize_kernel=True)
        noise = _pixel_noise(band_cfg, mt.pixel_scale_arcsec)
        if band in (EuclidBand.VIS, EuclidBand.NIR_H):
            detected |= model > mt.segmap_threshold_sigma * noise

        folder = out_dir / "MER" / str(tile) / band.instrument_folder
        folder.mkdir(parents=True, exist_ok=True)
        header = tile_wcs.to_header()
        header["MAGZERO"] = band_cfg.magzero
        image = (model + rng.normal(0.0, noise, model.shape)).astype(np.float32)
        fits.PrimaryHDU(image, header=header).writeto(
            folder / f"EUC_MER_BGSUB-MOSAIC-{band.value}_TILE{tile}-MOCK.fits", overwrite=True
        )
        _write_psf_grid(folder / f"EUC_MER_GRID-PSF-{band.value}_TILE{tile}-MOCK.fits", psf, tile_wcs, mt)

    segmap, n_objects = ndi_label(detected)
    fits.PrimaryHDU(segmap.astype(np.int32), header=tile_wcs.to_header()).writeto(
        out_dir / "MER" / str(tile) / f"EUC_MER_FINAL-SEGMAP_TILE{tile}-MOCK.fits", overwrite=True
    )
    er = mt.empty_regions
    regions = find_empty_regions(
        segmap,
        tile_wcs,
        tile,
        min_empty_pixels_area=er.min_empty_pixels_area,
        min_object_distance=er.min_object_distance_px,
        min_border_distance=er.min_border_distance_px,
        peak_min_distance=er.peak_min_distance_px,
    )
    catalog = write_empty_region_catalog(regions, out_dir / "EmptyRegionsCatalog", tile)
    logger.info(f"mock tile {tile}: {n_objects} segments, {len(regions)} empty regions -> {catalog}")
    return out_dir


def synthetic_renders(n_pixels: int, size_arcmin: float) -> dict[EuclidBand, np.ndarray]:
    """Return noise-free mag/arcsec^2 renders of a synthetic interacting pair.

    Stands in for the TNG API output so the instrument chain can run without
    credentials: an exponential disc in the centre and a smaller, redder
    bulge-like companion. Same format as a TNG ``vis.hdf5`` grid.

    Args:
        n_pixels (int): Render side in pixels.
        size_arcmin (float): Render side in arcmin.

    Returns:
        dict[EuclidBand, np.ndarray]: Surface brightness per band.
    """
    pixel_arcsec = 60.0 * size_arcmin / n_pixels
    y, x = np.mgrid[:n_pixels, :n_pixels]
    c = n_pixels / 2
    disc = Sersic2D(1.0, r_eff=0.05 * n_pixels, n=1.0, x_0=c, y_0=c, ellip=0.4, theta=0.6)(x, y)
    companion = Sersic2D(1.0, r_eff=0.02 * n_pixels, n=2.5, x_0=c + 0.12 * n_pixels, y_0=c - 0.08 * n_pixels)(x, y)
    disc, companion = disc / disc.sum(), companion / companion.sum()
    total_mag = {
        EuclidBand.VIS: (20.5, 21.9),
        EuclidBand.NIR_Y: (20.1, 21.2),
        EuclidBand.NIR_J: (19.9, 20.9),
        EuclidBand.NIR_H: (19.8, 20.7),
    }
    renders = {}
    for band, (m_disc, m_comp) in total_mag.items():
        flux_px = JANSKY_AB_ZEROPOINT * (10 ** (-0.4 * m_disc) * disc + 10 ** (-0.4 * m_comp) * companion)
        renders[band] = -2.5 * np.log10(flux_px / pixel_arcsec**2 / JANSKY_AB_ZEROPOINT)
    return renders
