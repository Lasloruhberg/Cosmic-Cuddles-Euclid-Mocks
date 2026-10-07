"""Create the ScienceReady mock of one TNG subhalo (needs TNG credentials)."""

import argparse

from euclid_mocks.config import DEFAULT_CONFIG, configure_logging, load_config
from euclid_mocks.pipeline import galaxy_rng, make_mock


def main() -> None:
    """Parse arguments, run the pipeline for one subhalo and print its summary row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub-id", type=int, required=True, help="Subfind ID")
    parser.add_argument("--snapshot", type=int, required=True, help="Snapshot number")
    parser.add_argument("--merger-type", default="manual", help="Label used in the file name")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML run configuration")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg)

    # Merger times are unknown outside the merger catalogue; they are stored as undefined.
    sample = {
        "subhalo_id": args.sub_id,
        "merger_type": args.merger_type,
        "time_since_last_major": None,
        "time_since_last_minor": None,
        "time_until_next_major": None,
        "time_until_next_minor": None,
    }
    row = make_mock(cfg, sample, args.snapshot, galaxy_rng(cfg.sampling.seed, args.snapshot, args.sub_id))
    for key, value in row.items():
        print(f"{key:24s} {value}")


if __name__ == "__main__":
    main()
