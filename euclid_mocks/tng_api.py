"""Talk to the IllustrisTNG web API and cache what it returns on disk.

Every network access of the package goes through this module: subhalo and
snapshot metadata (memoised in memory), SubLink progenitor walks, ``vis.hdf5``
renders and merger-history files (both cached on disk).
"""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from dotmap import DotMap
from loguru import logger

from euclid_mocks.config import load_api_key
from euclid_mocks.constants import TNG_H, EuclidBand


class RenderTimeout(RuntimeError):
    """Signal that the render server's gateway gave up on a render (HTTP 504).

    With ``method=sphMap`` this happens for galaxies in very massive FoF groups,
    because every particle of the group is loaded. The pipeline catches it to
    switch to the configured fallback render method.
    """


class TNGRequestError(RuntimeError):
    """Signal that every attempt of a TNG API request failed.

    Keeps one entry per attempt, so the summary CSV shows the whole history
    (e.g. ``403 Forbidden`` followed by ``504 Gateway Time-out``) instead of
    only the last error. Messages carry status codes and reasons but no URLs,
    because the render server's redirect URLs contain an access token.

    Attributes:
        failures (list[str]): One description per failed attempt.
        status_codes (list[int | None]): HTTP status per attempt (``None`` for network errors).
    """

    def __init__(self, failures: list[str], status_codes: list[int | None]):
        super().__init__("; ".join(failures))
        self.failures = failures
        self.status_codes = status_codes


def _describe(exc: requests.RequestException) -> tuple[str, int | None]:
    if exc.response is not None:
        return f"HTTP {exc.response.status_code} {exc.response.reason}", exc.response.status_code
    return f"{type(exc).__name__}", None


def _request(
    url: str, api_key: str, timeout: float, n_attempts: int, retry_wait: float, params=None, stream=False
):
    """Send an authenticated GET and retry transient failures.

    The TNG render server occasionally drops requests under load, so a request
    is retried up to ``n_attempts`` times, waiting ``retry_wait`` seconds
    between attempts to give a busy server time to recover. If all attempts
    fail, the errors of all of them are raised together.

    Args:
        url (str): Absolute API URL.
        api_key (str): TNG API key.
        timeout (float): Per-request timeout in seconds.
        n_attempts (int): Total number of tries.
        retry_wait (float): Seconds to wait before each retry.
        params (dict | None): Query parameters.
        stream (bool): Stream the body instead of loading it into memory.

    Returns:
        requests.Response: Successful response.

    Raises:
        TNGRequestError: If every attempt fails.
    """
    failures, status_codes = [], []
    for attempt in range(1, n_attempts + 1):
        try:
            response = requests.get(
                url, params=params, headers={"api-key": api_key}, timeout=timeout, stream=stream
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            description, status = _describe(exc)
            failures.append(f"attempt {attempt}/{n_attempts}: {description}")
            status_codes.append(status)
            logger.warning(f"TNG request failed ({description}) for {url.split('?')[0]}")
            if attempt < n_attempts:
                time.sleep(retry_wait)
    raise TNGRequestError(failures, status_codes)


@lru_cache(maxsize=8192)
def _get_json_cached(url: str, api_key: str, timeout: float, n_attempts: int, retry_wait: float) -> dict:
    return _request(url, api_key, timeout, n_attempts, retry_wait).json()


def get_json(cfg: DotMap, url: str) -> dict:
    """Fetch a JSON API endpoint, memoised per process.

    Snapshot and subhalo records are requested repeatedly during a run (size,
    selection, metadata), so caching them removes most of the API traffic.

    Args:
        cfg (DotMap): Run configuration.
        url (str): Absolute API URL.

    Returns:
        dict: Decoded JSON response.
    """
    return _get_json_cached(url, load_api_key(cfg), cfg.tng.timeout_s, cfg.tng.n_attempts, cfg.tng.retry_wait_s)


def simulation_url(cfg: DotMap) -> str:
    """Return the API root of the configured simulation.

    All endpoint URLs of this module are built on top of it.

    Args:
        cfg (DotMap): Run configuration.

    Returns:
        str: URL such as ``https://www.tng-project.org/api/TNG50-1/``.
    """
    return f"{cfg.tng.base_url}{cfg.tng.simulation}/"


def get_snapshot(cfg: DotMap, snapshot: int) -> dict:
    """Return the API record of one snapshot.

    Used to look up the redshift when converting physical sizes to arcmin.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.

    Returns:
        dict: Snapshot record including ``redshift``.
    """
    return get_json(cfg, f"{simulation_url(cfg)}snapshots/{snapshot}/")


def get_snapshots(cfg: DotMap) -> list[dict]:
    """Return the API records of all snapshots of the simulation.

    The merger-catalogue stage turns these into lookback times.

    Args:
        cfg (DotMap): Run configuration.

    Returns:
        list[dict]: One record per snapshot with ``number`` and ``redshift``.
    """
    return get_json(cfg, f"{simulation_url(cfg)}snapshots/")


def get_subhalo(cfg: DotMap, snapshot: int, sub_id: int) -> dict:
    """Return the full API record of one subhalo.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.
        sub_id (int): Subfind ID within that snapshot.

    Returns:
        dict: Subhalo record (masses in 1e10 Msun/h, radii in ckpc/h).
    """
    return get_json(cfg, f"{simulation_url(cfg)}snapshots/{snapshot}/subhalos/{sub_id}/")


def query_subhalos(cfg: DotMap, snapshot: int, params: dict) -> dict:
    """Run a filtered subhalo list query on one snapshot.

    Used by the merger-catalogue stage to get all subhaloes above the stellar
    mass limit in a single request.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.
        params (dict): API filter parameters, e.g. ``{"mass_stars__gt": 1.0}``.

    Returns:
        dict: API list response with ``count`` and ``results``.
    """
    url = f"{simulation_url(cfg)}snapshots/{snapshot}/subhalos/"
    return _request(
        url, load_api_key(cfg), cfg.tng.timeout_s, cfg.tng.n_attempts, cfg.tng.retry_wait_s, params
    ).json()


def max_past_masses(cfg: DotMap, sub: dict) -> tuple[float, float]:
    """Return the maximum DM and stellar mass along the SubLink main progenitor branch.

    Walks at most ``cfg.mergers.past_mass_max_steps`` progenitor hops; this
    covers the recent merger window, which is all the analysis needs. The
    result is stored in the per-galaxy CSV row as a merger mass-ratio proxy.

    Args:
        cfg (DotMap): Run configuration.
        sub (dict): Subhalo record to start from.

    Returns:
        tuple[float, float]: ``(log10 M_dm, log10 M_star)`` maxima in Msun.
    """
    mass_dm, mass_stars = [sub["mass_dm"]], [sub["mass_stars"]]
    for _ in range(cfg.mergers.past_mass_max_steps):
        if sub["prog_sfid"] == -1:
            break
        sub = get_json(cfg, sub["related"]["sublink_progenitor"])
        mass_dm.append(sub["mass_dm"])
        mass_stars.append(sub["mass_stars"])
    with np.errstate(divide="ignore"):
        log_dm = np.log10(np.array(mass_dm) * 1e10 / TNG_H)
        log_stars = np.log10(np.array(mass_stars) * 1e10 / TNG_H)
    return float(np.nanmax(log_dm)), float(np.nanmax(log_stars))


def render_url(
    cfg: DotMap, snapshot: int, sub_id: int, band: EuclidBand, size_arcmin: float, method: str
) -> str:
    """Build the ``vis.hdf5`` URL of a surface-brightness render.

    ``method=sphMap`` renders all particles of the parent FoF group with their
    SPH kernel, which keeps companions and tidal features of merging systems
    visible; ``sphMap_subhalo`` renders only the subhalo's own particles.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.
        sub_id (int): Subfind ID.
        band (EuclidBand): Euclid band to render.
        size_arcmin (float): Side length of the render in arcmin.
        method (str): API render method, e.g. ``sphMap`` or ``sphMap_subhalo``.

    Returns:
        str: Request URL.
    """
    r = cfg.render
    return (
        f"{simulation_url(cfg)}snapshots/{snapshot}/subhalos/{sub_id}/vis.hdf5"
        f"?partType={r.particle_type}&partField={r.particle_field}{band.tng_filter}"
        f"&size={size_arcmin}&sizeType=arcmin&nPixels={r.n_pixels}"
        f"&method={method}&axes={r.axes}"
    )


def download_render(
    cfg: DotMap,
    snapshot: int,
    sub_id: int,
    band: EuclidBand,
    size_arcmin: float,
    method: str,
    n_attempts: int,
) -> Path:
    """Download (or reuse) the surface-brightness render of one subhalo in one band.

    Renders are cached per render setting and method, so re-running a
    snapshot never hits the render server twice for the same image.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.
        sub_id (int): Subfind ID.
        band (EuclidBand): Euclid band to render.
        size_arcmin (float): Side length of the render in arcmin.
        method (str): API render method.
        n_attempts (int): Total tries for this request.

    Returns:
        Path: Path to the cached HDF5 file (dataset ``grid`` in mag/arcsec^2).

    Raises:
        RenderTimeout: If the last attempt ended with HTTP 504 (message lists all attempts).
        TNGRequestError: If the request failed for another reason.
    """
    r = cfg.render
    folder = (
        Path(cfg.paths.tng_cache_dir)
        / f"pix{r.n_pixels}-ax{r.axes}-siz{size_arcmin}_arcmin_{band.tng_filter}_{method}"
    )
    path = folder / f"grid_subhalo_{cfg.tng.simulation}_{snapshot}_{sub_id}.hdf5"
    if path.exists():
        logger.debug(f"render cached: {path}")
        return path
    url = render_url(cfg, snapshot, sub_id, band, size_arcmin, method)
    logger.debug(f"requesting {url}")
    try:
        response = _request(url, load_api_key(cfg), cfg.tng.timeout_s, n_attempts, cfg.tng.retry_wait_s)
    except TNGRequestError as exc:
        if exc.status_codes[-1] == 504:
            raise RenderTimeout(
                f"{method} {band.value} render of sub {sub_id} (snap {snapshot}) timed out [{exc}]"
            ) from exc
        raise
    folder.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.rename(path)
    return path


def download_merger_history(cfg: DotMap, snapshot: int) -> Path:
    """Download (or reuse) the Eisert et al. (2023) merger-history file of a snapshot.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.

    Returns:
        Path: Local HDF5 path.
    """
    path = Path(cfg.paths.merger_dir) / f"MergerHistory_{snapshot:03d}.hdf5"
    if path.exists():
        return path
    url = f"{simulation_url(cfg)}files/merger_history.{snapshot}.hdf5"
    logger.info(f"downloading merger history {url}")
    response = _request(
        url, load_api_key(cfg), cfg.tng.timeout_s, cfg.tng.n_attempts, cfg.tng.retry_wait_s, stream=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name so an interrupted download is never mistaken for a cache hit.
    partial = path.with_suffix(".part")
    with open(partial, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    partial.rename(path)
    return path
