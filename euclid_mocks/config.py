"""Load the YAML run configuration into a DotMap."""

import sys
from pathlib import Path

import yaml
from dotmap import DotMap
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> DotMap:
    """Read a YAML configuration and return it as a strict DotMap.

    Every module receives this object and reads parameters as attributes, e.g.
    ``cfg.render.n_pixels``. Entries under ``paths`` that are relative are
    resolved against the directory of the YAML file, so a config can be moved
    together with its data. The DotMap is non-dynamic: a typo in a key raises
    instead of silently creating an empty entry.

    Args:
        path (str | Path): YAML file to load.

    Returns:
        DotMap: Configuration with absolute path strings under ``paths``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)
    for key, value in raw["paths"].items():
        raw["paths"][key] = str((path.parent / value).resolve())
    return DotMap(raw, _dynamic=False)


def config_from_dict(raw: dict) -> DotMap:
    """Rebuild a strict DotMap from ``cfg.toDict()`` inside a worker process.

    Worker processes receive the configuration as a plain dict because that
    pickles reliably under the ``spawn`` start method used on macOS.

    Args:
        raw (dict): Output of ``DotMap.toDict()``.

    Returns:
        DotMap: Strict configuration identical to the parent's.
    """
    return DotMap(raw, _dynamic=False)


def load_api_key(cfg: DotMap) -> str:
    """Read the TNG API key from the credentials file named in the config.

    Called lazily by :mod:`euclid_mocks.tng_api` so that offline runs on the
    mock tile do not need credentials.

    Args:
        cfg (DotMap): Run configuration.

    Returns:
        str: The TNG API key.

    Raises:
        FileNotFoundError: If the credentials file is missing.
    """
    credentials = Path(cfg.paths.credentials_file)
    if not credentials.exists():
        raise FileNotFoundError(
            f"{credentials} not found. Copy config/tng_credentials.example.yaml to "
            "config/tng_credentials.yaml and add your TNG API key "
            "(https://www.tng-project.org/users/profile/)."
        )
    with open(credentials) as f:
        return yaml.safe_load(f)["api_key"]


def configure_logging(cfg: DotMap) -> None:
    """Route loguru output to stderr at ``cfg.run.log_level``.

    Called by the scripts and by every worker process; spawned workers would
    otherwise log at loguru's default DEBUG level.

    Args:
        cfg (DotMap): Run configuration.
    """
    logger.remove()
    logger.add(sys.stderr, level=cfg.run.log_level)
