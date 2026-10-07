"""Run the mock-imaging pipeline for one galaxy or for whole snapshots.

``make_mock_from_renders`` is the instrument part (renders -> ScienceReady
file) and needs no network. ``make_mock`` adds the TNG part (metadata,
selection, render download). ``run_snapshot`` samples galaxies from the
merger catalogue and processes them in parallel.
"""

import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from dotmap import DotMap
from loguru import logger

from euclid_mocks import tng_api
from euclid_mocks.background import choose_background, inject_background
from euclid_mocks.bands import observe_nisp, observe_vis
from euclid_mocks.config import config_from_dict, configure_logging
from euclid_mocks.constants import ALL_BANDS, NISP_BANDS, EuclidBand
from euclid_mocks.merger_catalog import load_or_build_merger_catalog
from euclid_mocks.science_ready import (
    BandProduct,
    build_science_ready,
    find_existing,
    output_path,
    read_detection_levels,
)
from euclid_mocks.source import (
    RejectedGalaxy,
    check_render_brightness,
    check_subhalo_selection,
    load_render,
    render_size_arcmin,
    surface_brightness_to_flux_hdu,
)

# Merger-dict keys mapped to the CSV columns the downstream analysis reads.
_TIME_COLUMNS = {
    "time_to_last_major": "time_since_last_major",
    "time_to_last_minor": "time_since_last_minor",
    "time_to_next_major": "time_until_next_major",
    "time_to_next_minor": "time_until_next_minor",
}


def galaxy_rng(seed: int, snapshot: int, sub_id: int) -> np.random.Generator:
    """Return the random generator of one galaxy.

    Seeding from ``(seed, snapshot, sub_id)`` makes every mock reproducible and
    independent of which worker process handles it.

    Args:
        seed (int): Run seed.
        snapshot (int): Snapshot number.
        sub_id (int): Subfind ID.

    Returns:
        np.random.Generator: Generator for noise and background draws.
    """
    return np.random.default_rng([seed, snapshot, sub_id])


def mock_header_cards(
    cfg: DotMap, sample: dict, snapshot: int, size_arcmin: float, sub: dict | None, render_method: str
) -> dict:
    """Build the ``M_*`` header cards that describe how a mock was made.

    Args:
        cfg (DotMap): Run configuration.
        sample (dict): Merger-catalogue entry (``merger_type`` and merger times).
        snapshot (int): Snapshot number.
        size_arcmin (float): Render side length.
        sub (dict | None): Subhalo record; ``None`` for synthetic sources.
        render_method (str): API render method actually used (or ``synthetic``).

    Returns:
        dict: Header keyword -> value.
    """
    r = cfg.render
    return {
        "M_VIEW": r.axes,
        "M_PIX": r.n_pixels,
        "M_SIZ": size_arcmin,
        "M_S_TYP": "arcmin",
        "M_SIM": cfg.tng.simulation,
        "M_SNAP": snapshot,
        "M_PT": r.particle_type,
        "M_PF": r.particle_field,
        "M_METH": render_method,
        "M_MERG": sample["merger_type"],
        "M_GALMA": None if sub is None else sub["mass"],
        "M_STELMA": None if sub is None else sub["mass_stars"],
        "M_TLMJ": sample["time_since_last_major"],
        "M_TLMI": sample["time_since_last_minor"],
        "M_TNMJ": sample["time_until_next_major"],
        "M_TNMI": sample["time_until_next_minor"],
    }


def make_mock_from_renders(
    cfg: DotMap,
    renders: dict[EuclidBand, np.ndarray],
    size_arcmin: float,
    sub_id: int,
    snapshot: int,
    header_cards: dict,
    rng: np.random.Generator,
) -> tuple[Path, list[float]]:
    """Turn per-band surface-brightness renders into a ScienceReady FITS file.

    Steps: true sky in Jy -> VIS stack (defines the grid) -> NISP stacks on
    that grid -> one background patch for all bands -> PSF convolution and
    background injection per band -> geometry-checked ScienceReady file.

    Args:
        cfg (DotMap): Run configuration.
        renders (dict[EuclidBand, np.ndarray]): mag/arcsec^2 render per band.
        size_arcmin (float): Side length of the renders.
        sub_id (int): Subfind ID (only used for naming).
        snapshot (int): Snapshot number (only used for naming).
        header_cards (dict): ``M_*`` cards from :func:`mock_header_cards`.
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        tuple[Path, list[float]]: Written file and detection levels (VIS, Y, J, H).
    """
    stacks = {EuclidBand.VIS: observe_vis(cfg, surface_brightness_to_flux_hdu(renders[EuclidBand.VIS], size_arcmin), rng)}
    for band in NISP_BANDS:
        flux = surface_brightness_to_flux_hdu(renders[band], size_arcmin)
        stacks[band] = observe_nisp(cfg, flux, band, stacks[EuclidBand.VIS], rng)

    position = choose_background(cfg, stacks[EuclidBand.VIS].data.shape, rng)
    products = {}
    for band in ALL_BANDS:
        header = stacks[band].header.copy()
        header.update(header_cards)
        header["M_FILT"] = band.tng_filter
        science, header, detection_level, noise = inject_background(
            cfg, stacks[band].data, header, band, position
        )
        products[band] = BandProduct(band, stacks[band].data, science, header, detection_level, noise)

    path = output_path(cfg, header_cards["M_MERG"], sub_id, snapshot, position.tile, position.index)
    build_science_ready(products).writeto(path, overwrite=True)
    return path, [products[band].detection_level for band in ALL_BANDS]


def _row(sample: dict, snapshot: int, status: str, **values) -> dict:
    row = {
        "subid": sample["subhalo_id"],
        "snapshot": snapshot,
        "merger_type": sample["merger_type"],
        "galaxy_mass": None,
        "stellar_mass": None,
        "max_dm_mass_past": None,
        "max_stellar_mass_past": None,
        **{column: sample[key] for column, key in _TIME_COLUMNS.items()},
        "hdu_path": None,
        "snr": None,
        "render_method": None,
        "status": status,
        "error": None,
    }
    row.update(values)
    return row


def _download_renders(
    cfg: DotMap, snapshot: int, sub_id: int, size_arcmin: float, method: str, n_attempts: int
) -> dict[EuclidBand, np.ndarray]:
    renders = {}
    for band in ALL_BANDS:
        path = tng_api.download_render(cfg, snapshot, sub_id, band, size_arcmin, method, n_attempts)
        renders[band] = load_render(path)
        # Faint objects are rejected before the three NISP renders are requested.
        if band is EuclidBand.VIS:
            check_render_brightness(cfg, renders[band])
    return renders


def fetch_renders(
    cfg: DotMap, snapshot: int, sub_id: int, size_arcmin: float
) -> tuple[dict[EuclidBand, np.ndarray], str]:
    """Download the four band renders of a subhalo, switching method on a render timeout.

    All bands are rendered with ``cfg.render.method``. If any of them times
    out (HTTP 504) and ``cfg.render.timeout_fallback.enabled`` is set, all four
    are rendered again with the fallback method, so the bands never mix
    methods. The primary method is tried only once when the fallback is
    enabled, because a timeout caused by a huge FoF group is deterministic.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.
        sub_id (int): Subfind ID.
        size_arcmin (float): Render side length.

    Returns:
        tuple: ``(renders per band in mag/arcsec^2, render method used)``.

    Raises:
        RenderTimeout: If the primary method times out and the fallback is disabled.
        TNGRequestError: If the fallback also fails; the message holds both errors.
        RejectedGalaxy: If the VIS render is too faint.
    """
    fallback = cfg.render.timeout_fallback
    primary_attempts = 1 if fallback.enabled else cfg.tng.n_attempts
    try:
        renders = _download_renders(cfg, snapshot, sub_id, size_arcmin, cfg.render.method, primary_attempts)
        return renders, cfg.render.method
    except tng_api.RenderTimeout as exc:
        if not fallback.enabled:
            raise
        primary_error = exc
        logger.warning(f"{exc}; falling back to {fallback.method} for all bands")
        # The timed-out render may still occupy the server; give it a moment before the next request.
        time.sleep(cfg.tng.retry_wait_s)
    try:
        renders = _download_renders(cfg, snapshot, sub_id, size_arcmin, fallback.method, cfg.tng.n_attempts)
    except (tng_api.TNGRequestError, tng_api.RenderTimeout) as exc:
        raise tng_api.TNGRequestError(
            [f"{primary_error}", f"fallback {fallback.method} failed: {exc}"], []
        ) from exc
    return renders, fallback.method


def make_mock(cfg: DotMap, sample: dict, snapshot: int, rng: np.random.Generator) -> dict:
    """Create (or find) the ScienceReady mock of one TNG subhalo and describe it.

    Fetches the subhalo, applies the selection cuts, downloads the renders via
    :func:`fetch_renders` (rejecting faint objects, falling back to the
    subhalo-only method on a render timeout if enabled), and runs
    :func:`make_mock_from_renders`. If a ScienceReady file for this object and
    view already exists, only its metadata row is rebuilt.

    Args:
        cfg (DotMap): Run configuration.
        sample (dict): Merger-catalogue entry with ``subhalo_id``, ``merger_type`` and times.
        snapshot (int): Snapshot number.
        rng (np.random.Generator): Per-galaxy random generator.

    Returns:
        dict: One CSV row (status ``created`` or ``existing``).

    Raises:
        RejectedGalaxy: If the subhalo fails a selection cut.
    """
    sub_id = sample["subhalo_id"]
    sub = tng_api.get_subhalo(cfg, snapshot, sub_id)
    max_dm, max_stars = tng_api.max_past_masses(cfg, sub)
    metadata = {
        "galaxy_mass": sub["mass"],
        "stellar_mass": sub["mass_stars"],
        "max_dm_mass_past": max_dm,
        "max_stellar_mass_past": max_stars,
    }

    existing = find_existing(cfg, sub_id, snapshot)
    if existing is not None:
        logger.info(f"sub {sub_id} snap {snapshot}: using existing {existing.name}")
        return _row(sample, snapshot, "existing", hdu_path=str(existing),
                    snr=_snr_string(read_detection_levels(existing)),
                    render_method=fits.getheader(existing, "VIS")["M_METH"], **metadata)

    redshift = tng_api.get_snapshot(cfg, snapshot)["redshift"]
    size_arcmin = render_size_arcmin(cfg, sub, redshift)
    check_subhalo_selection(cfg, sub, size_arcmin)

    renders, method = fetch_renders(cfg, snapshot, sub_id, size_arcmin)
    cards = mock_header_cards(cfg, sample, snapshot, size_arcmin, sub, method)
    path, levels = make_mock_from_renders(cfg, renders, size_arcmin, sub_id, snapshot, cards, rng)
    logger.info(f"sub {sub_id} snap {snapshot}: wrote {path.name} ({method})")
    return _row(sample, snapshot, "created", hdu_path=str(path), snr=_snr_string(levels),
                render_method=method, **metadata)


def _snr_string(levels: list[float]) -> str:
    # Downstream analysis parses this exact "[vis,y,j,h]" format.
    return "[" + ",".join(str(float(v)) for v in levels) + "]"


def process_sample(task: tuple[dict, dict, int]) -> dict:
    """Process one galaxy in a worker process and never let it abort the batch.

    This is the only place that catches exceptions: a rejected or failed
    galaxy is logged and recorded with its status and error message.

    Args:
        task (tuple[dict, dict, int]): ``(cfg.toDict(), sample, snapshot)``.

    Returns:
        dict: CSV row with status ``created``, ``existing``, ``rejected`` or ``failed``.
    """
    raw_cfg, sample, snapshot = task
    cfg = config_from_dict(raw_cfg)
    configure_logging(cfg)
    rng = galaxy_rng(cfg.sampling.seed, snapshot, sample["subhalo_id"])
    try:
        return make_mock(cfg, sample, snapshot, rng)
    except RejectedGalaxy as exc:
        logger.info(f"sub {sample['subhalo_id']} snap {snapshot} rejected: {exc}")
        return _row(sample, snapshot, "rejected", error=str(exc))
    except Exception as exc:
        logger.exception(f"sub {sample['subhalo_id']} snap {snapshot} failed")
        return _row(sample, snapshot, "failed", error=f"{type(exc).__name__}: {exc}")


def select_samples(cfg: DotMap, merger_dict: dict) -> list[dict]:
    """Draw the galaxies of one snapshot from its merger catalogue.

    Draws up to ``n_major`` majors, ``n_minor`` minors and ``n_non_merging``
    non-merging galaxies, in that order, from one ``random.Random(seed)``, so
    the selection matches the original ``random.seed(seed)`` behaviour.

    Args:
        cfg (DotMap): Run configuration.
        merger_dict (dict): Output of :func:`load_or_build_merger_catalog`.

    Returns:
        list[dict]: Selected merger-catalogue entries.
    """
    s = cfg.sampling
    draw = random.Random(s.seed)
    samples = []
    for key, n in (("major_mergers", s.n_major), ("minor_mergers", s.n_minor), ("non_merging", s.n_non_merging)):
        samples += draw.sample(merger_dict[key], min(len(merger_dict[key]), n))
    return samples


def run_snapshot(cfg: DotMap, snapshot: int) -> Path:
    """Create the mocks of one snapshot in parallel and write its summary CSV.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.

    Returns:
        Path: The written CSV (one row per sampled galaxy).
    """
    samples = select_samples(cfg, load_or_build_merger_catalog(cfg, snapshot))
    logger.info(f"snap {snapshot}: processing {len(samples)} galaxies")
    tasks = [(cfg.toDict(), sample, snapshot) for sample in samples]
    with Pool(processes=cfg.run.n_processes) as pool:
        rows = pool.map(process_sample, tasks)

    name = cfg.run.csv_name.format(
        version=cfg.run.version, sample_size=cfg.sampling.n_major, snapshot=snapshot, seed=cfg.sampling.seed
    )
    path = Path(cfg.paths.output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"snap {snapshot}: wrote {path}")
    return path
