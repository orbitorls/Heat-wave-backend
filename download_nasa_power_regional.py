"""
Download additional weather data from NASA POWER API - REGIONAL MODE
No API key required! Free historical data for Thailand region.

Uses the regional endpoint which returns a 0.5x0.5 degree grid in one request.

Variables available:
- PRECTOTCORR: Precipitation corrected (mm/day)
- RH2M: Relative Humidity at 2m (%)
- T2MDEW: Dewpoint Temperature at 2m (°C)
- T2M: Temperature at 2m (°C)

Coverage: 1981-present (daily)
"""

import os
import requests
import numpy as np
import xarray as xr
from datetime import datetime
import time
import pandas as pd

# Thailand bounding box
LAT_MIN = 5.0
LAT_MAX = 21.0
LON_MIN = 97.0
LON_MAX = 106.0

OUTPUT_DIR = "era5_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NASA POWER API regional endpoint
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/regional"


def download_nasa_power_regional(start_date, end_date):
    """
    Download NASA POWER data for entire Thailand region in one request.
    
    Returns DataFrame with multi-index (time, lat, lon)
    """
    params = {
        "community": "RE",
        "latitude-min": LAT_MIN,
        "latitude-max": LAT_MAX,
        "longitude-min": LON_MIN,
        "longitude-max": LON_MAX,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "parameters": "PRECTOTCORR,RH2M,T2MDEW,T2M",
    }
    
    print(f"Requesting regional data: {start_date} to {end_date}")
    print(f"Region: {LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E")
    
    try:
        # Build URL for display
        req = requests.Request('GET', POWER_API, params=params)
        print(f"URL: {req.prepare().url[:100]}...")
        
        response = requests.get(POWER_API, params=params, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text[:500]}")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None


def parse_regional_response(data):
    """
    Parse NASA POWER regional response into xarray Dataset.
    
    Returns xarray Dataset with dimensions (time, latitude, longitude)
    """
    if data is None:
        return None
    
    # Extract geometry (contains lat/lon arrays)
    geometry = data.get('geometry', {})
    coords = geometry.get('coordinates', [])
    
    if not coords:
        print("No coordinates in response")
        return None
    
    # For polygon, extract outer ring then lat/lon
    if geometry.get('type') == 'Polygon':
        # coords is [[[lon, lat], [lon, lat], ...]]
        ring = coords[0]  # outer ring
        lons = sorted(list(set([c[0] for c in ring])))
        lats = sorted(list(set([c[1] for c in ring])), reverse=True)
    else:
        print(f"Unknown geometry type: {geometry.get('type')}")
        return None
    
    print(f"Grid: {len(lats)} lats x {len(lons)} lons")
    
    # Extract parameter data
    properties = data.get('properties', {})
    parameter = properties.get('parameter', {})
    
    if not parameter:
        print("No parameters in response")
        return None
    
    # Get date range from first parameter
    first_param = list(parameter.values())[0]
    date_strs = sorted(first_param.keys())
    
    print(f"Date range: {date_strs[0]} to {date_strs[-1]}")
    print(f"Days: {len(date_strs)}")
    
    # Build 3D arrays
    ntime = len(date_strs)
    nlat = len(lats)
    nlon = len(lons)
    
    # Initialize arrays
    precip = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    rh = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    dew = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    temp = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    
    # Parse data - parameter keys are dates, values are 2D arrays (lat x lon)
    for ti, date_str in enumerate(date_strs):
        for param_name, param_data in parameter.items():
            values = param_data.get(date_str)
            
            if values is None:
                continue
                
            # Values is a list of lists (lat rows, lon cols)
            try:
                arr = np.array(values, dtype=np.float32)
                
                if param_name == 'PRECTOTCORR':
                    precip[ti] = arr
                elif param_name == 'RH2M':
                    rh[ti] = arr
                elif param_name == 'T2MDEW':
                    dew[ti] = arr
                elif param_name == 'T2M':
                    temp[ti] = arr
            except Exception as e:
                print(f"Error parsing {param_name} for {date_str}: {e}")
    
    # Create xarray Dataset
    times = [datetime.strptime(d, '%Y%m%d') for d in date_strs]
    
    ds = xr.Dataset(
        {
            'PRECTOT': (['time', 'latitude', 'longitude'], precip),
            'RH2M': (['time', 'latitude', 'longitude'], rh),
            'T2MDEW': (['time', 'latitude', 'longitude'], dew),
            'T2M': (['time', 'latitude', 'longitude'], temp),
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons,
        }
    )
    
    # Add attributes
    ds['PRECTOT'].attrs['units'] = 'mm/day'
    ds['RH2M'].attrs['units'] = '%'
    ds['T2MDEW'].attrs['units'] = '°C'
    ds['T2M'].attrs['units'] = '°C'
    
    return ds


def download_year(year):
    """Download one year of data."""
    output_file = f"{OUTPUT_DIR}/nasa_power_{year}.nc"
    
    if os.path.exists(output_file):
        print(f"Skip {year} (exists): {output_file}")
        return True
    
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    print(f"\n{'='*50}")
    print(f"Downloading year {year}...")
    print(f"{'='*50}")
    
    data = download_nasa_power_regional(start_date, end_date)
    
    if data:
        ds = parse_regional_response(data)
        if ds is not None:
            ds.to_netcdf(output_file)
            print(f"Saved: {output_file}")
            print(f"Shape: time={len(ds.time)}, lat={len(ds.latitude)}, lon={len(ds.longitude)}")
            return True
    
    return False


def test_regional():
    """Test regional API with one year."""
    print("Testing NASA POWER Regional API...")
    
    # Test with small date range first
    data = download_nasa_power_regional("20150101", "20150131")
    
    if data:
        ds = parse_regional_response(data)
        if ds is not None:
            print("\nSuccess! Dataset:")
            print(ds)
            print("\nSample values:")
            print(ds.isel(time=0).to_dataframe())
            return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download NASA POWER data for Thailand (Regional)")
    parser.add_argument("--start", type=int, default=2000, help="Start year")
    parser.add_argument("--end", type=int, default=2015, help="End year")
    parser.add_argument("--test", action="store_true", help="Test API with small sample")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NASA POWER Regional Data Downloader")
    print("=" * 60)
    print(f"Area: Thailand ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)")
    print(f"Period: {args.start}-{args.end}")
    print(f"Variables: PRECTOTCORR, RH2M, T2MDEW, T2M")
    print(f"Resolution: 0.5 x 0.5 degrees (regional API)")
    print()
    
    if args.test:
        test_regional()
    else:
        success = 0
        failed = 0
        
        for year in range(args.start, args.end + 1):
            if download_year(year):
                success += 1
            else:
                failed += 1
                print(f"Failed: {year}")
        
        print(f"\n{'='*50}")
        print(f"Complete: {success} years, {failed} failed")
        print(f"Files: {OUTPUT_DIR}/nasa_power_YYYY.nc")