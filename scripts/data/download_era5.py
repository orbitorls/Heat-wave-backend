import argparse
import datetime as dt
import os

import cdsapi


MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
DEFAULT_TIMES = ["00:00", "06:00", "12:00", "18:00"]
DEFAULT_AREA = [21, 97, 5, 106]  # [North, West, South, East]


def _parse_months(raw):
    raw = (raw or "all").strip().lower()
    if raw in {"all", "*"}:
        return MONTHS

    selected = set()
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            start_m = int(start_s)
            end_m = int(end_s)
            if start_m > end_m:
                start_m, end_m = end_m, start_m
            for month in range(start_m, end_m + 1):
                if 1 <= month <= 12:
                    selected.add(f"{month:02d}")
        else:
            month = int(piece)
            if 1 <= month <= 12:
                selected.add(f"{month:02d}")

    if not selected:
        raise ValueError("No valid months provided.")
    return sorted(selected)


def _parse_times(raw):
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_TIMES
    times = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        hh_mm = value.split(":")
        if len(hh_mm) != 2:
            raise ValueError(f"Invalid time format: {value}")
        hour = int(hh_mm[0])
        minute = int(hh_mm[1])
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError(f"Invalid time value: {value}")
        times.append(f"{hour:02d}:{minute:02d}")
    if not times:
        raise ValueError("No valid times provided.")
    return sorted(set(times))


def _parse_area(raw):
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_AREA
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("Area must be 4 comma-separated numbers: north,west,south,east")
    return [float(v) for v in parts]


def _iter_year_month(start_year, end_year, months):
    for year in range(start_year, end_year + 1):
        for month in months:
            yield year, month


def _download_surface(cds_client, output_path, year, month, times, area):
    cds_client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "2m_temperature",
                "volumetric_soil_water_layer_1",
                "total_precipitation",
                "2m_dewpoint_temperature",
            ],
            "year": str(year),
            "month": month,
            "day": DAYS,
            "time": times,
            "area": area,
        },
        output_path,
    )


def _download_upper(cds_client, output_path, year, month, times, area):
    cds_client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": "geopotential",
            "pressure_level": "500",
            "year": str(year),
            "month": month,
            "day": DAYS,
            "time": times,
            "area": area,
        },
        output_path,
    )


def download_era5_data(
    start_year,
    end_year,
    months,
    times,
    area,
    output_dir,
    force=False,
):
    os.makedirs(output_dir, exist_ok=True)
    cds_client = cdsapi.Client()

    print("ERA5 download started")
    print(f"Years: {start_year}-{end_year}")
    print(f"Months: {','.join(months)}")
    print(f"Times per day: {','.join(times)}")
    print(f"Area [N,W,S,E]: {area}")
    print(f"Output dir: {output_dir}")

    for year, month in _iter_year_month(start_year, end_year, months):
        ym = f"{year}{month}"
        surface_filename = os.path.join(output_dir, f"era5_surface_plus_{ym}.nc")
        upper_filename = os.path.join(output_dir, f"era5_upper_plus_{ym}.nc")

        print(f"\n[{ym}]")

        if os.path.exists(surface_filename) and not force:
            print(f"  Skip surface (exists): {surface_filename}")
        else:
            print("  Download surface levels...")
            try:
                _download_surface(cds_client, surface_filename, year, month, times, area)
                print(f"  Saved: {surface_filename}")
            except Exception as exc:
                print(f"  Surface download failed: {exc}")

        if os.path.exists(upper_filename) and not force:
            print(f"  Skip upper (exists): {upper_filename}")
        else:
            print("  Download pressure levels...")
            try:
                _download_upper(cds_client, upper_filename, year, month, times, area)
                print(f"  Saved: {upper_filename}")
            except Exception as exc:
                print(f"  Upper download failed: {exc}")

    print("\nERA5 download completed.")


def main():
    current_year = dt.datetime.now().year
    parser = argparse.ArgumentParser(
        description="Download ERA5 data for Thailand heatwave training."
    )
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=current_year - 1)
    parser.add_argument(
        "--months",
        type=str,
        default="all",
        help="Comma list or range, e.g. 'all', '1-12', '3,4,5'",
    )
    parser.add_argument(
        "--times",
        type=str,
        default=",".join(DEFAULT_TIMES),
        help="Comma list in HH:MM, e.g. '00:00,06:00,12:00,18:00'",
    )
    parser.add_argument(
        "--area",
        type=str,
        default="21,97,5,106",
        help="north,west,south,east",
    )
    parser.add_argument("--output-dir", type=str, default="era5_data")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing files.",
    )
    args = parser.parse_args()

    if args.end_year < args.start_year:
        raise ValueError("end-year must be greater than or equal to start-year")

    months = _parse_months(args.months)
    times = _parse_times(args.times)
    area = _parse_area(args.area)

    download_era5_data(
        start_year=args.start_year,
        end_year=args.end_year,
        months=months,
        times=times,
        area=area,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
