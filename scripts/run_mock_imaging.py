"""Create mocks for every snapshot in cfg.run.snapshots (needs TNG credentials).

For each snapshot: build the merger catalogue, sample galaxies, create their
ScienceReady files in parallel and write one summary CSV.
"""

import argparse

from euclid_mocks.config import DEFAULT_CONFIG, configure_logging, load_config
from euclid_mocks.pipeline import run_snapshot


def main() -> None:
    """Parse arguments and process the configured snapshots in turn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML run configuration")
    parser.add_argument("--snapshots", type=int, nargs="+", help="Override cfg.run.snapshots")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg)
    for snapshot in args.snapshots or cfg.run.snapshots:
        run_snapshot(cfg, snapshot)


if __name__ == "__main__":
    main()
