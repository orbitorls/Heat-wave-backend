"""
Download additional weather data from NASA POWER API
No API key required! Free historical data for Thailand region.

Variables available:
- PRECTOT: Precipitation (mm/day)
- RH2M: Relative Humidity at 2m (%)
- T2MDEW: Dewpoint Temperature at 2m (°C)
- T2M: Temperature at 2m (°C)

Coverage: 1981-present (daily), 2001-present (hourly)
"""

import os
import requests
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import time
import pandas as pd

# Thailand bounding box
LAT_MIN = 5.0
LAT_MAX = 21.0
LON_MIN = 97.0
LON_MAX = 106.0

# Grid resolution (0.5 degrees for NASA POWER)
LAT_STEP = 0.5
LON_STEP = 0.5

OUTPUT_DIR = "era5_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NASA POWER API base
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Community parameter for renewable energy (includes all met variables)
COMMUNITY = "RE"


def generate_grid_points():
    """Generate grid points for Thailand region."""
    lats = np.arange(LAT_MAX, LAT_MIN - LAT_STEP, -LAT_STEP)
    lons = np.arange(LON_MIN, LON_MAX + LON_STEP, LON_STEP)
    return lats, lons


def download_nasa_power(lat, lon, start_date, end_date):
    """
    Download NASA POWER data for a single point.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude  
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
    
    Returns:
    --------
    dict or None
        Dictionary with parameter data or None if failed
    """
    # Use 'parameters' instead of 'tslen', and specify actual parameters
    params = {
        "community": COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "parameters": "PRECTOT,RH2M,T2MDEW,T2M",
    }
    
    try:
        response = requests.get(POWER_API, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code} at ({lat}, {lon}): {response.text[:200]}")
            return None
    except Exception as e:
        print(f"Exception at ({lat}, {lon}): {e}")
        return None


def parse_power_response(data, lat, lon):
    """
    Parse NASA POWER response into DataFrame.
    
    Returns DataFrame with columns: time, lat, lon, PRECTOT, RH2M, T2MDEW, T2M
    """
    if data is None or 'properties' not in data:
        return None
    
    props = data['properties']
    parameters = props.get('parameter', {})
    
    if not parameters:
        return None
    
    # Extract dates and values - data is directly in the parameter dict
    dates = []
    precip = []
    rh = []
    dew = []
    temp = []
    
    # Get date range from first parameter
    first_param = list(parameters.values())[0]
    date_strs = list(first_param.keys())
    
    for date_str in date_strs:
        dates.append(datetime.strptime(date_str, '%Y%m%d'))
        
        # Extract values for each variable (direct dict, no 'data' key)
        p = parameters.get('PRECTOTCORR', parameters.get('PRECTOT', {})).get(date_str)
        h = parameters.get('RH2M', {}).get(date_str)
        d = parameters.get('T2MDEW', {}).get(date_str)
        t = parameters.get('T2M', {}).get(date_str)
        
        precip.append(p if p is not None else np.nan)
        rh.append(h if h is not None else np.nan)
        dew.append(d if d is not None else np.nan)
        temp.append(t if t is not None else np.nan)
    
    df = pd.DataFrame({
        'time': dates,
        'latitude': lat,
        'longitude': lon,
        'PRECTOT': precip,
        'RH2M': rh,
        'T2MDEW': dew,
        'T2M': temp,
    })
    
    return df


def download_all_grids(start_year, end_year):
    """
    Download NASA POWER data for all grid points in Thailand region.
    Saves as NetCDF file.
    """
    lats, lons = generate_grid_points()
    
    print(f"Grid: {len(lats)} latitudes x {len(lons)} longitudes = {len(lats)*len(lons)} points")
    print(f"Period: {start_year}-{end_year}")
    
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    
    all_data = []
    total = len(lats) * len(lons)
    count = 0
    
    for lat in lats:
        for lon in lons:
            count += 1
            
            # Download data
            data = download_nasa_power(lat, lon, start_date, end_date)
            
            if data:
                df = parse_power_response(data, lat, lon)
                if df is not None and len(df) > 0:
                    all_data.append(df)
            
            # Progress
            if count % 10 == 0:
                print(f"Progress: {count}/{total} ({(count/total)*100:.1f}%)")
            
            # Rate limiting - NASA POWER is generous but be polite
            time.sleep(0.2)
    
    if not all_data:
        print("No data downloaded!")
        return None
    
    # Combine all grid points
    print("Combining data from all grid points...")
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert to xarray Dataset
    ds = combined.set_index(['time', 'latitude', 'longitude']).to_xarray()
    
    return ds


def download_yearly(start_year, end_year, output_prefix="nasa_power"):
    """
    Download data year by year to avoid long requests and allow resuming.
    """
    for year in range(start_year, end_year + 1):
        output_file = f"{OUTPUT_DIR}/{output_prefix}_{year}.nc"
        
        if os.path.exists(output_file):
            print(f"Skip {year} (exists): {output_file}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Downloading year {year}...")
        print(f"{'='*50}")
        
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        lats, lons = generate_grid_points()
        
        all_data = []
        total = len(lats) * len(lons)
        count = 0
        
        for lat in lats:
            for lon in lons:
                count += 1
                
                data = download_nasa_power(lat, lon, start_date, end_date)
                
                if data:
                    df = parse_power_response(data, lat, lon)
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                
                if count % 20 == 0:
                    print(f"Progress: {count}/{total} ({(count/total)*100:.1f}%)")
                
                time.sleep(0.15)  # Rate limit
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            ds = combined.set_index(['time', 'latitude', 'longitude']).to_xarray()
            
            # Save as NetCDF
            ds.to_netcdf(output_file)
            print(f"Saved: {output_file}")
            print(f"Records: {len(combined)}, Grid points: {len(lats)*len(lons)}")
        else:
            print(f"No data for year {year}")


def test_single_point():
    """Test API with single point."""
    print("Testing NASA POWER API...")
    
    # Bangkok area
    lat, lon = 13.75, 100.5
    start_date = "20150101"
    end_date = "20150110"
    
    data = download_nasa_power(lat, lon, start_date, end_date)
    
    if data:
        df = parse_power_response(data, lat, lon)
        if df is not None:
            print("Success! Sample data:")
            print(df.head())
            return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download NASA POWER data for Thailand")
    parser.add_argument("--start", type=int, default=2000, help="Start year")
    parser.add_argument("--end", type=int, default=2015, help="End year")
    parser.add_argument("--test", action="store_true", help="Test API with single point")
    parser.add_argument("--output", default="nasa_power", help="Output file prefix")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NASA POWER Historical Weather Data Downloader")
    print("=" * 60)
    print(f"Area: Thailand ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)")
    print(f"Period: {args.start}-{args.end}")
    print(f"Variables: PRECTOT, RH2M, T2MDEW, T2M")
    print()
    
    if args.test:
        test_single_point()
    else:
        download_yearly(args.start, args.end, args.output)
        print("\nDownload complete!")
        print(f"Files saved to: {OUTPUT_DIR}/nasa_power_YYYY.nc")