"""
Download additional weather data from Open-Meteo API
This is a free API that provides historical weather data including:
- Temperature
- Precipitation
- Humidity
- Dewpoint

No API key required!
"""

import os
import requests
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import time

# Thailand bounding box
LAT_MIN = 5.0
LAT_MAX = 21.0
LON_MIN = 97.0
LON_MAX = 106.0

# Grid resolution (approximate 0.25 degrees like ERA5)
LAT_STEP = 0.25
LON_STEP = 0.25

OUTPUT_DIR = "era5_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_grid_points(lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    """Generate grid points for the area."""
    lats = np.arange(lat_max, lat_min - lat_step, -lat_step)
    lons = np.arange(lon_min, lon_max + lon_step, lon_step)
    return lats, lons


def download_open_meteo_data(start_date, end_date, output_file):
    """Download historical weather data from Open-Meteo API."""
    
    # Generate grid points
    lats, lons = generate_grid_points(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, LAT_STEP, LON_STEP)
    
    print(f"Grid: {len(lats)} latitudes x {len(lons)} longitudes")
    
    # Variables to download
    variables = [
        "temperature_2m",
        "relative_humidity_2m", 
        "dewpoint_2m",
        "precipitation",
    ]
    
    all_data = []
    
    # Download data for each grid point (Open-Meteo allows multiple points)
    # We'll download in batches to avoid overwhelming the API
    
    # Create time range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"Downloading data from {start_date} to {end_date}...")
    print(f"Total days: {len(dates)}")
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            url = "https://archive-api.open-meteo.com/v1/archive"
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ",".join(variables),
                "timezone": "Asia/Bangkok",
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if "hourly" in data:
                        hourly_data = data["hourly"]
                        
                        # Create DataFrame
                        df = pd.DataFrame(hourly_data)
                        df["latitude"] = lat
                        df["longitude"] = lon
                        all_data.append(df)
                        
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error at ({lat}, {lon}): {e}")
                continue
            
            # Progress
            if (i * len(lons) + j + 1) % 10 == 0:
                print(f"Progress: {i * len(lons) + j + 1}/{len(lats) * len(lons)}")
    
    if all_data:
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Save as CSV
        combined.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        print(f"Total records: {len(combined)}")
        
        return combined
    else:
        print("No data downloaded")
        return None


def download_era5_land():
    """Alternative: Download ERA5-Land data (requires CDS API)."""
    print("""
To download ERA5-Land data (more complete):
1. Register at https://cds.climate.copernicus.eu/
2. Install CDS API key
3. Run: python download_era5.py --start 2000 --end 2015

ERA5-Land includes:
- 2m temperature
- 2m dewpoint temperature  
- Total precipitation
- Soil moisture
- Snow depth
- And more...
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download additional weather data")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2015-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="open_meteo_data.csv", help="Output filename")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Open-Meteo Historical Weather Data Downloader")
    print("=" * 60)
    print(f"Area: Thailand ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)")
    print(f"Period: {args.start} to {args.end}")
    print()
    
    # Check if data already exists
    output_path = os.path.join(OUTPUT_DIR, args.output)
    if os.path.exists(output_path):
        print(f"Data file already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download")
            exit(0)
    
    # Download data
    result = download_open_meteo_data(args.start, args.end, output_path)
    
    if result is not None:
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        print("\nVariables available:")
        for col in result.columns:
            print(f"  - {col}")
    else:
        print("Download failed")