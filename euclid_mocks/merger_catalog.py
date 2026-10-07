"""Classify TNG subhaloes of a snapshot as major, minor, non-merging or unclear.

Uses the merger-history catalogue of Eisert et al. (2023): for every subhalo
it holds the snapshot of the last and next major and minor merger. These are
turned into time differences and compared with configurable windows. The
result is the merger dictionary that the sampling stage draws from.
"""

import pickle
from pathlib import Path

import h5py
import numpy as np
from dotmap import DotMap
from loguru import logger

from euclid_mocks import tng_api
from euclid_mocks.constants import TNG_COSMOLOGY, TNG_H

MERGER_CLASSES = ("major_mergers", "minor_mergers", "non_merging", "unclear")


def lookback_times(cfg: DotMap) -> dict[int, float]:
    """Return the lookback time in Gyr of every snapshot of the simulation.

    Args:
        cfg (DotMap): Run configuration.

    Returns:
        dict[int, float]: Snapshot number -> lookback time.
    """
    return {
        snap["number"]: TNG_COSMOLOGY.lookback_time(snap["redshift"]).value
        for snap in tng_api.get_snapshots(cfg)
    }


def massive_subhalo_ids(cfg: DotMap, snapshot: int) -> set[int]:
    """Return the IDs of all subhaloes above the stellar-mass limit (query cached on disk).

    The API query pre-filters on stellar mass (converted to 1e10 Msun/h with
    the TNG h) and on a minimum half-mass radius; the exact
    ``mass_log_msun >= limit`` cut is then applied to the result.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.

    Returns:
        set[int]: Subfind IDs.
    """
    m = cfg.mergers
    cache = Path(cfg.paths.merger_dir) / (
        f"snapnum_{snapshot}_subslim_{m.import_limit}_min{m.load_mass_limit_log_msun}_minmax.pkl"
    )
    if cache.exists():
        with open(cache, "rb") as f:
            subs = pickle.load(f)
    else:
        params = {
            "limit": m.import_limit,
            "mass_stars__gt": 10**m.load_mass_limit_log_msun / 1e10 * TNG_H,
            "mass_stars__lt": 10**m.max_mass_log_msun / 1e10 * TNG_H,
            "halfmassrad_stars__gt": m.min_halfmassrad_stars,
        }
        subs = tng_api.query_subhalos(cfg, snapshot, params)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(subs, f)
    ids = {s["id"] for s in subs["results"] if s["mass_log_msun"] >= m.load_mass_limit_log_msun}
    logger.info(f"snap {snapshot}: {len(ids)} of {subs['count']} queried subhaloes above the mass limit")
    return ids


def classify(times: dict[str, float | None], snaps: dict[str, int], cfg: DotMap) -> str:
    """Assign the merger class of one subhalo from its merger time differences.

    A subhalo is ``major`` if its last or next major merger is within the
    major window, else ``minor`` if its last or next minor merger is within
    the minor window, else ``non_merging`` if it never merges or all four
    mergers lie beyond the non-merging window, and ``unclear`` otherwise.

    Args:
        times (dict[str, float | None]): Absolute time differences in Gyr
            (``None`` = no such merger), keyed ``last_major``, ``next_major``,
            ``last_minor``, ``next_minor``.
        snaps (dict[str, int]): Corresponding merger snapshots (-1 = none).
        cfg (DotMap): Run configuration.

    Returns:
        str: One of :data:`MERGER_CLASSES`.
    """
    m = cfg.mergers

    def within(key: str, window: float) -> bool:
        return times[key] is not None and times[key] <= window

    if within("last_major", m.major_window_gyr) or within("next_major", m.major_window_gyr):
        return "major_mergers"
    if within("last_minor", m.minor_window_gyr) or within("next_minor", m.minor_window_gyr):
        return "minor_mergers"
    never_merges = all(s == -1 for s in snaps.values())
    all_far = all(t is None or t > m.non_merging_window_gyr for t in times.values())
    if never_merges or all_far:
        return "non_merging"
    return "unclear"


_LABEL = {"major_mergers": "major", "minor_mergers": "minor", "non_merging": "non-merging", "unclear": "unclear"}


def build_merger_catalog(cfg: DotMap, snapshot: int) -> dict[str, list[dict]]:
    """Classify all subhaloes above the mass limit of one snapshot.

    Subhaloes are visited in a seeded random order and classification stops
    once both ``max_major`` majors and ``max_minor`` minors are collected, so
    large snapshots do not require a full pass.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.

    Returns:
        dict[str, list[dict]]: Entries per class in :data:`MERGER_CLASSES`.
    """
    m = cfg.mergers
    valid_ids = massive_subhalo_ids(cfg, snapshot)
    lookback = lookback_times(cfg)
    now = lookback[snapshot]

    with h5py.File(tng_api.download_merger_history(cfg, snapshot), "r") as f:
        group = f["WithConstraint"] if m.with_constraint else f
        columns = {
            "mass_last_major": group["MassLastMajorMerger"][:],
            "snap_last_major": group["SnapNumLastMajorMerger"][:],
            "snap_next_major": group["SnapNumNextMajorMerger"][:],
            "snap_last_minor": group["SnapNumLastMinorMerger"][:],
            "snap_next_minor": group["SnapNumNextMinorMerger"][:],
        }
    n_subhalos = len(columns["mass_last_major"])
    indices = np.array(sorted(i for i in valid_ids if i < n_subhalos), dtype=np.int64)
    indices = np.random.default_rng(m.shuffle_seed).permutation(indices)

    result = {key: [] for key in MERGER_CLASSES}
    for i in indices:
        if len(result["major_mergers"]) >= m.max_major and len(result["minor_mergers"]) >= m.max_minor:
            break
        snaps = {key: int(columns[f"snap_{key}"][i]) for key in ("last_major", "next_major", "last_minor", "next_minor")}
        times = {key: None if s == -1 else float(abs(now - lookback[s])) for key, s in snaps.items()}
        merger_class = classify(times, snaps, cfg)
        result[merger_class].append(
            {
                "subhalo_id": int(i),
                "mass_last_major": float(columns["mass_last_major"][i]),
                "snap_last_major": snaps["last_major"],
                "time_since_last_major": times["last_major"],
                "snap_next_major": snaps["next_major"],
                "time_until_next_major": times["next_major"],
                "snap_last_minor": snaps["last_minor"],
                "time_since_last_minor": times["last_minor"],
                "snap_next_minor": snaps["next_minor"],
                "time_until_next_minor": times["next_minor"],
                "merger_type": _LABEL[merger_class],
            }
        )
    logger.info(f"snap {snapshot}: " + ", ".join(f"{len(v)} {k}" for k, v in result.items()))
    return result


def load_or_build_merger_catalog(cfg: DotMap, snapshot: int) -> dict[str, list[dict]]:
    """Return the merger dictionary of a snapshot, building and pickling it once.

    Args:
        cfg (DotMap): Run configuration.
        snapshot (int): Snapshot number.

    Returns:
        dict[str, list[dict]]: Entries per class in :data:`MERGER_CLASSES`.
    """
    path = Path(cfg.paths.merger_dir) / f"merger_events_{snapshot}_{cfg.mergers.load_mass_limit_log_msun}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    catalog = build_merger_catalog(cfg, snapshot)
    with open(path, "wb") as f:
        pickle.dump(catalog, f)
    return catalog
