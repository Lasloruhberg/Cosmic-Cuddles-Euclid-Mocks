"""Build the synthetic MER tile and its empty-region catalogue under data/mock_tile/."""

import argparse

from euclid_mocks.config import DEFAULT_CONFIG, configure_logging, load_config
from euclid_mocks.mock_tile import build_mock_tile


def main() -> None:
    """Parse arguments and write the mock tile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML run configuration")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg)
    build_mock_tile(cfg)


if __name__ == "__main__":
    main()
