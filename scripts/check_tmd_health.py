"""Manual health check for TMD API — do NOT run in CI (hits live API, burns quota).

Usage:
    python scripts/check_tmd_health.py

Requires TMD_API_KEY in environment or .env file.
"""
import asyncio
import sys
import os
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.data.tmd_client import TMDClient, TMDAPIError, TMDAuthError


async def main() -> None:
    client = TMDClient()
    yesterday = date.today() - timedelta(days=1)
    station = "BKK_01"
    print(f"Checking TMD API for station={station} date={yesterday}...")
    try:
        obs = await client.fetch_hourly(station, yesterday)
        print(f"  OK — received {len(obs)} observations")
        if obs:
            first = obs[0]
            print(f"  Sample: {first.ts_utc}  temp={first.temp_c}°C  rh={first.rh}%")
    except TMDAuthError:
        print("  FAIL — Invalid or missing TMD_API_KEY. Check your .env file.")
        sys.exit(1)
    except TMDAPIError as e:
        print(f"  FAIL — {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
