import type { Station } from "./api-types";

// Fallback mirror of app/data/stations.py — used when /stations endpoint is unavailable
export const STATIONS_FALLBACK: Station[] = [
  { station_id: "BKK_01", name_th: "กรุงเทพมหานคร (Don Mueang)", lat: 13.9132, lon: 100.6067, elevation_m: 9.5 },
  { station_id: "CNX_01", name_th: "เชียงใหม่", lat: 18.7761, lon: 98.9769, elevation_m: 310.0 },
  { station_id: "KKN_01", name_th: "ขอนแก่น", lat: 16.4419, lon: 102.8359, elevation_m: 182.0 },
  { station_id: "HYI_01", name_th: "หาดใหญ่ (สงขลา)", lat: 6.9269, lon: 100.437, elevation_m: 8.0 },
  { station_id: "RYG_01", name_th: "ระยอง", lat: 12.6815, lon: 101.2816, elevation_m: 14.0 },
];
