"""End-to-end check of the geometry guarantee: every ScienceReady extension shares VIS's shape and WCS."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_mocks.config import load_config
from euclid_mocks.constants import ALL_BANDS
from euclid_mocks.mock_tile import build_mock_tile, synthetic_renders
from euclid_mocks.pipeline import galaxy_rng, make_mock_from_renders, mock_header_cards
from euclid_mocks.science_ready import assert_same_geometry

EXPECTED_EXTENSIONS = [b.value for b in ALL_BANDS] + [f"{b.value}DET" for b in ALL_BANDS]


@pytest.fixture(scope="module")
def science_ready_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    cfg = load_config()
    cfg.paths.output_dir = str(root / "output")
    cfg.paths.euclid_mer_dir = str(root / "tile" / "MER")
    cfg.paths.empty_region_dir = str(root / "tile" / "EmptyRegionsCatalog")
    cfg.mock_tile.size_px = 512
    build_mock_tile(cfg, root / "tile")

    size_arcmin = 0.15
    sample = {
        "merger_type": "synthetic",
        "time_since_last_major": None,
        "time_since_last_minor": None,
        "time_until_next_major": None,
        "time_until_next_minor": None,
    }
    path, _ = make_mock_from_renders(
        cfg,
        synthetic_renders(cfg.render.n_pixels, size_arcmin),
        size_arcmin,
        sub_id=0,
        snapshot=99,
        header_cards=mock_header_cards(cfg, sample, 99, size_arcmin, None, "synthetic"),
        rng=galaxy_rng(cfg.sampling.seed, 99, 0),
    )
    return path


def test_all_extensions_share_vis_shape_and_wcs(science_ready_file):
    with fits.open(science_ready_file) as hdul:
        assert [hdu.name for hdu in hdul[1:]] == EXPECTED_EXTENSIONS
        vis = hdul["VIS"]
        vis_wcs = WCS(vis.header)
        for name in EXPECTED_EXTENSIONS:
            hdu = hdul[name]
            assert hdu.data.shape == vis.data.shape, name
            wcs = WCS(hdu.header)
            np.testing.assert_allclose(wcs.wcs.crpix, vis_wcs.wcs.crpix, err_msg=name)
            np.testing.assert_allclose(wcs.wcs.crval, vis_wcs.wcs.crval, err_msg=name)
            np.testing.assert_allclose(wcs.pixel_scale_matrix, vis_wcs.pixel_scale_matrix, err_msg=name)
            assert np.all(np.isfinite(hdu.data)), name


def test_geometry_guard_rejects_shifted_wcs(science_ready_file):
    with fits.open(science_ready_file) as hdul:
        vis = hdul["VIS"]
        shifted = fits.ImageHDU(data=vis.data, header=vis.header.copy(), name="NIR-Y")
        shifted.header["CRPIX1"] += 0.5
        with pytest.raises(ValueError, match="CRPIX"):
            assert_same_geometry(vis, shifted)
