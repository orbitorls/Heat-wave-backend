"""
Download additional weather data from NASA POWER API - CHUNKED MODE
No API key required! Free historical data for Thailand region.

Downloads in chunks to respect API limits:
- Max 1 parameter per request
- Max 10 degree latitude range

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

# Thailand bounding box
LAT_MIN = 5.0
LAT_MAX = 21.0
LON_MIN = 97.0
LON_MAX = 106.0

OUTPUT_DIR = "era5_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NASA POWER API regional endpoint
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/regional"

# API limits
MAX_LAT_RANGE = 10.0
MAX_LON_RANGE = 10.0  # not mentioned but let's be safe
MAX_PARAMS = 1


def generate_chunks():
    """Generate lat/lon chunks for downloading."""
    chunks = []
    
    # Split latitude into chunks of MAX_LAT_RANGE
    lat = LAT_MIN
    while lat < LAT_MAX:
        lat_end = min(lat + MAX_LAT_RANGE, LAT_MAX)
        chunks.append({
            'lat_min': lat,
            'lat_max': lat_end,
            'lon_min': LON_MIN,
            'lon_max': LON_MAX,
            'name': f"{int(lat*100)}_{int(lat_end*100)}N"
        })
        lat = lat_end
    
    print(f"Generated {len(chunks)} latitude chunks")
    return chunks


def download_chunk(lat_min, lat_max, lon_min, lon_max, start_date, end_date, param):
    """Download one chunk for one parameter."""
    params = {
        "community": "RE",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "parameters": param,
    }
    
    try:
        response = requests.get(POWER_API, params=params, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"  Exception: {e}")
        return None


def parse_chunk_response(data, param):
    """Parse chunk response into arrays (GeoJSON features format)."""
    if data is None:
        return None, None, None
    
    # GeoJSON features format
    features = data.get('features', [])
    
    if not features:
        return None, None, None
    
    # Extract all unique lats and lons
    lats = sorted(list(set([f['geometry']['coordinates'][1] for f in features])), reverse=True)
    lons = sorted(list(set([f['geometry']['coordinates'][0] for f in features])))
    
    # Get date range from first feature
    first_props = features[0].get('properties', {})
    param_data = first_props.get('parameter', {}).get(param, {})
    
    if not param_data:
        return None, None, None
    
    dates = sorted(param_data.keys())
    
    ntime = len(dates)
    nlat = len(lats)
    nlon = len(lons)
    
    print(f"    Grid: {nlat} lats x {nlon} lons, {ntime} days")
    
    # Create mapping from (lat, lon) to feature data
    # Note: coordinates are [lon, lat, elevation]
    feature_map = {}
    for f in features:
        coords = f['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        point_data = f.get('properties', {}).get('parameter', {}).get(param, {})
        feature_map[(lat, lon)] = point_data
    
    # Build array
    values = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)
    
    for lat_idx, lat in enumerate(lats):
        for lon_idx, lon in enumerate(lons):
            point_data = feature_map.get((lat, lon), {})
            
            for ti, date_str in enumerate(dates):
                val = point_data.get(date_str)
                if val is not None:
                    # -999 is fill value
                    if val != -999:
                        values[ti, lat_idx, lon_idx] = val
    
    return values, lats, lons


def download_year_chunked(year, chunks):
    """Download one year in chunks."""
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    # Parameters to download (one at a time) - Weather + Wind + Pressure (no radiation - different resolution)
    params_list = [
        ('PRECTOTCORR', 'mm/day'),  # Precipitation
        ('RH2M', '%'),               # Humidity
        ('T2MDEW', '°C'),           # Dewpoint
        ('T2M', '°C'),              # Temperature
        ('WS10M', 'm/s'),           # Wind speed at 10m
        ('PS', 'kPa'),              # Surface pressure
    ]
    
    # Store results for each parameter - collect all lats from all chunks
    param_results = {}  # param -> (values, lats, lons)
    all_chunk_lats = {}  # param -> list of lats arrays
    all_chunk_lons = {}  # param -> list of lons arrays
    
    for param, _ in params_list:
        print(f"\n  Downloading {param}...")
        
        all_values = []
        param_chunk_lats = []
        param_chunk_lons = []
        
        for i, chunk in enumerate(chunks):
            print(f"    Chunk {i+1}/{len(chunks)}: {chunk['lat_min']}-{chunk['lat_max']}N")
            
            data = download_chunk(
                chunk['lat_min'], chunk['lat_max'],
                chunk['lon_min'], chunk['lon_max'],
                start_date, end_date, param
            )
            
            values, lats, lons = parse_chunk_response(data, param)
            
            if values is not None:
                all_values.append(values)
                param_chunk_lats.append(lats)
                param_chunk_lons.append(lons)
            
            # Rate limit
            time.sleep(0.5)
        
        if all_values:
            # Concatenate along lat axis - but need to handle overlapping lats
            # Chunks overlap at chunk boundaries, so remove duplicate lats from subsequent chunks
            
            # Get reference lats from first chunk (include all)
            combined = all_values[0].copy()
            combined_lats = list(param_chunk_lats[0])  # copy
            combined_lons = param_chunk_lons[0]  # reference
            
            # For subsequent chunks, find overlapping lats and remove them
            for ci in range(1, len(all_values)):
                chunk_vals = all_values[ci]
                chunk_lats = param_chunk_lats[ci]
                
                # Find overlap - find lats that are in both combined_lats and chunk_lats
                # Overlap is at the high lat end
                overlap_count = 0
                for lat in chunk_lats:
                    if lat in combined_lats:
                        overlap_count += 1
                    else:
                        break  # assume sorted descending
                
                if overlap_count > 0:
                    # Remove overlap from this chunk's data
                    chunk_vals = chunk_vals[:, overlap_count:, :]
                    chunk_lats = chunk_lats[overlap_count:]
                
                # Concatenate
                combined = np.concatenate([combined, chunk_vals], axis=1)
                combined_lats.extend(chunk_lats)
            
            param_results[param] = (combined, combined_lats, combined_lons)
            print(f"    {param}: shape {combined.shape}, lats={len(combined_lats)}, lons={len(combined_lons)}")
    
    return param_results


def save_year_netcdf(year, param_results):
    """Save param results as NetCDF."""
    if not param_results:
        return False
    
    # Get dimensions from first param
    first_param = list(param_results.values())[0]
    values, lats, lons = first_param
    
    ntime = values.shape[0]
    times = [datetime(year, 1, 1) + timedelta(days=i) for i in range(ntime)]
    
    # Build dataset with consistent coordinates
    # Use the coordinates from each parameter (they should all be the same)
    all_lats = lats
    all_lons = lons
    
    data_vars = {}
    for param, (arr, lats_p, lons_p) in param_results.items():
        # Use the parameter's own coordinates
        data_vars[param] = (['time', 'latitude', 'longitude'], arr)
        if all_lats is lats:  # first param
            all_lats = lats_p
            all_lons = lons_p
    
    ds = xr.Dataset(
        data_vars,
        coords={
            'time': times,
            'latitude': all_lats,
            'longitude': all_lons,
        }
    )
    
    # Add units
    ds['PRECTOTCORR'].attrs['units'] = 'mm/day'
    ds['RH2M'].attrs['units'] = '%'
    ds['T2MDEW'].attrs['units'] = '°C'
    ds['T2M'].attrs['units'] = '°C'
    
    output_file = f"{OUTPUT_DIR}/nasa_power_{year}.nc"
    ds.to_netcdf(output_file)
    print(f"Saved: {output_file}")
    
    return True


def download_year(year, chunks):
    """Download one year of data."""
    output_file = f"{OUTPUT_DIR}/nasa_power_{year}.nc"
    
    if os.path.exists(output_file):
        print(f"Skip {year} (exists)")
        return True
    
    print(f"\n{'='*50}")
    print(f"Year {year}")
    print(f"{'='*50}")
    
    param_results = download_year_chunked(year, chunks)
    
    if param_results:
        return save_year_netcdf(year, param_results)
    
    return False


def download_multi_year(start_year, end_year, chunks):
    """Download multiple years - optimize by combining years where possible."""
    for year in range(start_year, end_year + 1):
        if download_year(year, chunks):
            # Success - continue
            pass
        else:
            print(f"Failed: {year}")
            # Continue with next year
    print(f"\nDownload complete for {start_year}-{end_year}")


def test_single_chunk():
    """Test with single chunk and parameter."""
    print("Testing NASA POWER chunked API...")
    
    # Single chunk, single param
    chunks = [{'lat_min': 5.0, 'lat_max': 15.0, 'lon_min': 97.0, 'lon_max': 106.0, 'name': 'test'}]
    
    data = download_chunk(5.0, 15.0, 97.0, 106.0, "20150101", "20150110", "T2M")
    
    if data:
        values, lats, lons = parse_chunk_response(data, "T2M")
        if values is not None:
            print(f"Success! Shape: {values.shape}")
            print(f"Lats: {lats}")
            print(f"Lons: {lons}")
            return True
    
    return False


if __name__ == "__main__":
    import argparse
    from datetime import timedelta
    
    parser = argparse.ArgumentParser(description="Download NASA POWER data for Thailand")
    parser.add_argument("--start", type=int, default=2000, help="Start year")
    parser.add_argument("--end", type=int, default=2000, help="End year")
    parser.add_argument("--test", action="store_true", help="Test API")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NASA POWER Chunked Downloader")
    print("=" * 60)
    print(f"Area: Thailand ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)")
    print(f"Period: {args.start}-{args.end}")
    print(f"Variables: PRECTOTCORR, RH2M, T2MDEW, T2M")
    print()
    
    if args.test:
        test_single_chunk()
    else:
        chunks = generate_chunks()
        
        success = 0
        failed = 0
        
        for year in range(args.start, args.end + 1):
            if download_year(year, chunks):
                success += 1
            else:
                failed += 1
                print(f"Failed: {year}")
        
        print(f"\n{'='*50}")
        print(f"Complete: {success} years, {failed} failed")
        print(f"Files: {OUTPUT_DIR}/nasa_power_YYYY.nc")