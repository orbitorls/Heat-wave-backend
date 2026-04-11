import { useCallback, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

import Overview from './components/Overview';
import { Forecast } from './components/Forecast';
import MapComponent from './components/Map';
import {
  type HeatLevel,
  extractTemperatureFromPayload,
  normalizeForecastEntry,
  normalizeRisk,
} from './shared/heatNormalization';
import './App.css';

interface Region {
  name: string;
  zone: string;
  lat: number;
  lng: number;
  temperature: number;
  heatLevel: HeatLevel;
}

interface ForecastDay {
  day: number;
  date: string;
  day_name: string;
  temperature: number;
  risk_level: HeatLevel;
  probability: number | null;
}

interface RegionForecastPayload {
  name: string;
  forecast?: unknown[];
}

type DataStatus = 'offline' | 'model-unavailable' | 'live-global' | 'api-error';
type ActiveTab = 'overview' | 'forecast' | 'map';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000').replace(/\/+$/, '');

const THAILAND_REGIONS = [
  { name: 'Bangkok', lat: 13.7563, lng: 100.5018, zone: 'Central' },
  { name: 'Chiang Mai', lat: 18.7883, lng: 98.9853, zone: 'North' },
  { name: 'Chiang Rai', lat: 19.9072, lng: 99.8329, zone: 'North' },
  { name: 'Khon Kaen', lat: 16.4423, lng: 102.1426, zone: 'Northeast' },
  { name: 'Nakhon Si Thammarat', lat: 8.4333, lng: 99.9333, zone: 'South' },
  { name: 'Surat Thani', lat: 9.1401, lng: 99.3331, zone: 'South' },
  { name: 'Pattaya', lat: 12.9333, lng: 100.8833, zone: 'Central' },
  { name: 'Phitsanulok', lat: 16.8281, lng: 100.2624, zone: 'North' },
  { name: 'Korat', lat: 14.9799, lng: 102.0782, zone: 'Northeast' },
];

const ThermometerIcon = ({ className = '' }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
  </svg>
);

const ActivityIcon = ({ className = '' }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
  </svg>
);

const LayoutDashboardIcon = ({ className = '' }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <rect width="7" height="9" x="3" y="3" rx="1"/>
    <rect width="7" height="5" x="14" y="3" rx="1"/>
    <rect width="7" height="9" x="14" y="12" rx="1"/>
    <rect width="7" height="5" x="3" y="16" rx="1"/>
  </svg>
);

const CalendarDaysIcon = ({ className = '' }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <rect width="18" height="18" x="3" y="4" rx="2" ry="2"/>
    <line x1="16" y1="2" x2="16" y2="6"/>
    <line x1="8" y1="2" x2="8" y2="6"/>
    <line x1="3" y1="10" x2="21" y2="10"/>
    <path d="M8 14h.01"/>
    <path d="M12 14h.01"/>
    <path d="M16 14h.01"/>
    <path d="M8 18h.01"/>
    <path d="M12 18h.01"/>
    <path d="M16 18h.01"/>
  </svg>
);

const MapIcon = ({ className = '' }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/>
    <line x1="9" y1="3" x2="9" y2="18"/>
    <line x1="15" y1="6" x2="15" y2="21"/>
  </svg>
);

const getStatusLabel = (status: DataStatus): string => {
  if (status === 'live-global') return 'ข้อมูลรายพื้นที่ล่าสุดจากระบบ (Live)';
  if (status === 'model-unavailable') return 'ยังไม่มีข้อมูลล่าสุด (Unavailable)';
  if (status === 'api-error') return 'โหลดข้อมูลล่าสุดไม่สำเร็จ (Error)';
  return 'ยังไม่เชื่อมต่อข้อมูล (Offline)';
};

const isLiveStatus = (status: DataStatus): boolean => status === 'live-global';

const toForecastRows = (payload: unknown): unknown[] => {
  if (!payload || typeof payload !== 'object') return [];
  const body = payload as Record<string, unknown>;
  if (Array.isArray(body.forecasts)) return body.forecasts;
  if (Array.isArray(body.forecast)) return body.forecast;
  return [];
};

const toRegionRows = (payload: unknown): unknown[] => {
  if (!payload || typeof payload !== 'object') return [];
  const body = payload as Record<string, unknown>;
  if (Array.isArray(body.regions)) return body.regions;
  return [];
};

const toRegionForecastRows = (payload: unknown): RegionForecastPayload[] => {
  if (!payload || typeof payload !== 'object') return [];
  const body = payload as Record<string, unknown>;
  if (!Array.isArray(body.region_forecasts)) return [];
  return body.region_forecasts.filter(
    (entry): entry is RegionForecastPayload => !!entry && typeof entry === 'object' && 'name' in entry
  );
};

export default function App() {
  const [regions, setRegions] = useState<Region[]>([]);
  const [forecast, setForecast] = useState<ForecastDay[]>([]);
  const [regionForecasts, setRegionForecasts] = useState<Record<string, ForecastDay[]>>({});
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [forecastRegion, setForecastRegion] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [loading, setLoading] = useState(true);
  const [dataStatus, setDataStatus] = useState<DataStatus>('offline');
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const fetchData = useCallback(async () => {
    try {
      const healthResponse = await fetch(`${API_BASE_URL}/api/health`).catch(() => null);
      if (!healthResponse?.ok) {
        setDataStatus('offline');
        setRegions([]);
        setForecast([]);
        setRegionForecasts({});
        return;
      }

      const healthData = (await healthResponse.json().catch(() => ({}))) as Record<string, unknown>;
      const modelLoaded = healthData.model_loaded !== false;

      if (!modelLoaded) {
        setDataStatus('model-unavailable');
        setRegions([]);
        setForecast([]);
        setRegionForecasts({});
        setLastUpdated(new Date().toLocaleTimeString());
        return;
      }

      const [predictResult, forecastResult] = await Promise.allSettled([
        fetch(`${API_BASE_URL}/api/predict`),
        fetch(`${API_BASE_URL}/api/forecast`),
      ]);

      let nextRegions: Region[] = [];
      let nextForecast: ForecastDay[] = [];
      let nextRegionForecasts: Record<string, ForecastDay[]> = {};
      let hasApiData = false;

      if (predictResult.status === 'fulfilled' && predictResult.value.ok) {
        const payload = (await predictResult.value.json().catch(() => null)) as unknown;
        const regionRows = toRegionRows(payload);

        if (regionRows.length > 0) {
          nextRegions = regionRows
            .map((row) => {
              const obj = row as Record<string, unknown>;
              const name = typeof obj.name === 'string' ? obj.name : '';
              const zone = typeof obj.zone === 'string' ? obj.zone : 'Central';
              const lat = typeof obj.lat === 'number' ? obj.lat : Number(obj.lat);
              const lng = typeof obj.lng === 'number' ? obj.lng : Number(obj.lng);
              const temperature = extractTemperatureFromPayload(obj) ?? (typeof obj.temperature_c === 'number' ? obj.temperature_c : Number(obj.temperature_c));
              const risk = normalizeRisk(obj.risk_level ?? obj.risk_code, temperature ?? null);

              if (!name || !Number.isFinite(lat) || !Number.isFinite(lng) || temperature === null) {
                return null;
              }

              return {
                name,
                zone,
                lat,
                lng,
                temperature,
                heatLevel: risk.heatLevel,
              } satisfies Region;
            })
            .filter((region): region is Region => region !== null);

          if (nextRegions.length > 0) {
            hasApiData = true;
          }
        }

        const temperature = extractTemperatureFromPayload(payload);

        if (nextRegions.length === 0 && temperature !== null) {
          const obj = payload as Record<string, unknown>;
          const risk = normalizeRisk(obj.risk_level, temperature);
          nextRegions = THAILAND_REGIONS.map((region) => ({
            ...region,
            temperature,
            heatLevel: risk.heatLevel,
          }));
          hasApiData = true;
        }
      }

      if (forecastResult.status === 'fulfilled' && forecastResult.value.ok) {
        const payload = (await forecastResult.value.json().catch(() => null)) as unknown;
        const rows = toForecastRows(payload);
        const regionRows = toRegionForecastRows(payload);

        nextForecast = rows
          .map((row, index) => normalizeForecastEntry(row, index + 1))
          .filter((row): row is NonNullable<typeof row> => row !== null)
          .map((row) => ({
            day: row.day,
            date: row.date,
            day_name: row.dayName,
            temperature: row.temperature,
            risk_level: row.risk.heatLevel,
            probability: row.probability,
          }));

        if (nextForecast.length > 0) {
          hasApiData = true;
        }

        if (regionRows.length > 0) {
          nextRegionForecasts = regionRows.reduce<Record<string, ForecastDay[]>>((acc, regionEntry) => {
            const normalizedRows = (Array.isArray(regionEntry.forecast) ? regionEntry.forecast : [])
              .map((row, index) => normalizeForecastEntry(row, index + 1))
              .filter((row): row is NonNullable<typeof row> => row !== null)
              .map((row) => ({
                day: row.day,
                date: row.date,
                day_name: row.dayName,
                temperature: row.temperature,
                risk_level: row.risk.heatLevel,
                probability: row.probability,
              }));

            if (regionEntry.name && normalizedRows.length > 0) {
              acc[regionEntry.name] = normalizedRows;
            }
            return acc;
          }, {});

          if (Object.keys(nextRegionForecasts).length > 0) {
            hasApiData = true;
          }
        }
      }

      setRegions(nextRegions);
      setForecast(nextForecast);
      setRegionForecasts(nextRegionForecasts);
      setForecastRegion((current) => {
        if (current && (nextRegionForecasts[current] || nextRegions.some((region) => region.name === current))) {
          return current;
        }

        const firstRegionWithForecast = Object.keys(nextRegionForecasts)[0];
        if (firstRegionWithForecast) return firstRegionWithForecast;

        return nextRegions[0]?.name ?? null;
      });
      setDataStatus(hasApiData ? 'live-global' : 'api-error');
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error fetching data', error);
      setDataStatus('api-error');
      setRegions([]);
      setForecast([]);
      setRegionForecasts({});
    } finally {
      setTimeout(() => setLoading(false), 800);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchData]);

  useEffect(() => {
    if (selectedRegion && regionForecasts[selectedRegion]) {
      setForecastRegion(selectedRegion);
    }
  }, [selectedRegion, regionForecasts]);

  if (loading) {
    return (
      <div className="agni-app loading-screen">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className="loader-content"
        >
          <motion.div
            animate={{
              scale: [1, 1.1, 1],
              rotate: [0, 2, -2, 0],
            }}
            transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
          >
            <ThermometerIcon className="loader-icon" />
          </motion.div>
          <motion.h1
            initial={{ y: 10, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="loader-title"
          >
            AGNI
          </motion.h1>
          <div className="loader-track">
            <motion.div
              className="loader-progress"
              initial={{ width: '0%' }}
              animate={{ width: '100%' }}
              transition={{ duration: 1.5, ease: 'circInOut' }}
            />
          </div>
        </motion.div>
      </div>
    );
  }

  const navItems: Array<{ id: ActiveTab; label: string; icon: ({ className }: { className?: string }) => React.JSX.Element }> = [
    { id: 'overview', label: 'ภาพรวม (Overview)', icon: LayoutDashboardIcon },
    { id: 'forecast', label: 'พยากรณ์ (Forecast)', icon: CalendarDaysIcon },
    { id: 'map', label: 'แผนที่ (Map)', icon: MapIcon },
  ];

  return (
    <div className="agni-app">
      <div className="ambient-bg" />

      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <ThermometerIcon className="logo-icon" />
            <span className="logo-text">AGNI</span>
          </div>
          <div className={`status-badge ${isLiveStatus(dataStatus) ? 'live' : 'demo'}`}>
            <ActivityIcon className="status-icon" />
            {getStatusLabel(dataStatus)}
          </div>
        </div>

        <nav className="header-nav">
          {navItems.map(({ id, label, icon: Icon }) => (
              <button 
                key={id} 
                onClick={() => setActiveTab(id)} 
                className={`nav-btn ${activeTab === id ? 'active' : ''}`}
              >
              <span className="nav-btn-content">
                <Icon className="nav-icon" />
                <span className="nav-label">{label}</span>
              </span>
              {activeTab === id && (
                <motion.div
                  layoutId="nav-indicator"
                  className="nav-indicator"
                  transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                />
              )}
            </button>
          ))}
        </nav>

        <div className="header-right">
          <div className="last-updated">
            <span className="time-label">อัปเดตเมื่อ (Updated)</span>
            <span className="time-value">{lastUpdated || '-'}</span>
          </div>
        </div>
      </header>

      <main className="app-main" data-tab={activeTab}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="tab-content"
          >
            {activeTab === 'overview' && (
              <Overview regions={regions} selectedRegion={selectedRegion} onSelectRegion={setSelectedRegion} />
            )}
            {activeTab === 'forecast' && (
              <Forecast
                forecast={forecastRegion && regionForecasts[forecastRegion] ? regionForecasts[forecastRegion] : forecast}
                regionName={forecastRegion}
                availableRegions={regions.map((region) => region.name)}
                onSelectRegion={setForecastRegion}
              />
            )}
            {activeTab === 'map' && (
              <MapComponent regions={regions} selectedRegion={selectedRegion} onSelectRegion={setSelectedRegion} />
            )}
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>© {new Date().getFullYear()} Agni — ระบบพยากรณ์คลื่นความร้อนประเทศไทย (Thailand Heatwave Forecasting System)</p>
        </div>
      </footer>
    </div>
  );
}
