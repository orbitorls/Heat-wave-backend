"""
PROJECT_CONTEXT v2 helper:
Download ERA5 extension range (default 2016-2025) and optionally print
reminder for NDVI extension workflow.
"""

import argparse

from download_era5 import (
    DEFAULT_AREA,
    DEFAULT_TIMES,
    MONTHS,
    download_era5_data,
)


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 extension dataset.")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", default="era5_data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Compatibility flag from PROJECT_CONTEXT.md (no-op reminder).",
    )
    args = parser.parse_args()

    download_era5_data(
        start_year=args.start_year,
        end_year=args.end_year,
        months=MONTHS,
        times=DEFAULT_TIMES,
        area=DEFAULT_AREA,
        output_dir=args.output_dir,
        force=args.force,
    )

    if args.update_config:
        print(
            "Reminder: update config/config.yaml and NDVI workflow after extension download."
        )


if __name__ == "__main__":
    main()
