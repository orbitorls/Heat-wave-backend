import React, { useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import { motion, AnimatePresence } from 'framer-motion';
import 'leaflet/dist/leaflet.css';
import './Map.css';
import {
  HEAT_ADVICE_TEXT,
  HEAT_HEX,
  ZONE_THAI,
  formatTemperature,
  normalizeRisk,
  type HeatLevel,
} from '../shared/heatNormalization';

export interface Region {
  name: string;
  zone: string;
  temperature: number;
  heatLevel: HeatLevel;
  lat: number;
  lng: number;
}

export interface MapProps {
  regions: Region[];
  selectedRegion: string | null;
  onSelectRegion: (name: string | null) => void;
}

const toSafeNumber = (value: unknown, fallback = 0): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const CloseIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const LocationIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
    <circle cx="12" cy="10" r="3" />
  </svg>
);

const AlertTriangleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

function MapFlyTo({ region }: { region: Region | undefined }) {
  const map = useMap();
  useEffect(() => {
    if (region) {
      map.flyTo([region.lat, region.lng], 9, { duration: 1.2, easeLinearity: 0.25 });
    } else {
      // Return to default bounds when no region is selected
      map.flyTo([13.5, 101.5], 6, { duration: 1 });
    }
  }, [region, map]);
  return null;
}

const MapComponent: React.FC<MapProps> = ({ regions, selectedRegion, onSelectRegion }) => {
  const [showDiscovery, setShowDiscovery] = useState(true);

  const activeRegionData = useMemo(
    () => regions.find((region) => region.name === selectedRegion),
    [regions, selectedRegion]
  );

  const activeRegionRisk = useMemo(() => {
    if (!activeRegionData) return null;
    const temperature = toSafeNumber(activeRegionData.temperature);
    return normalizeRisk(activeRegionData.heatLevel, temperature);
  }, [activeRegionData]);

  const activeHeatLevel: HeatLevel = activeRegionRisk?.heatLevel ?? 'caution';

  const highRiskRegions = useMemo(() => {
    return regions
      .filter((r) => {
        const risk = normalizeRisk(r.heatLevel, toSafeNumber(r.temperature));
        return risk.heatLevel === 'danger' || risk.heatLevel === 'severe' || risk.heatLevel === 'warning';
      })
      .sort((a, b) => toSafeNumber(b.temperature) - toSafeNumber(a.temperature))
      .slice(0, 5);
  }, [regions]);

  return (
    <div className="map-page-container">
      <div className="map-layout">
        
        {/* Discovery Panel (Floating Left) */}
        <AnimatePresence>
          {showDiscovery && (
            <motion.div 
              className="map-discovery-panel"
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -20, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            >
              <div className="discovery-header">
                <h3>จุดที่ต้องเฝ้าระวัง</h3>
                <span className="discovery-count">{highRiskRegions.length} พื้นที่</span>
              </div>
              
              {highRiskRegions.length === 0 ? (
                <div className="discovery-empty">
                  ไม่มีพื้นที่เสี่ยงสูงในขณะนี้
                </div>
              ) : (
                <div className="discovery-list">
                  {highRiskRegions.map((region) => {
                    const temp = toSafeNumber(region.temperature);
                    const risk = normalizeRisk(region.heatLevel, temp);
                    const color = HEAT_HEX[risk.heatLevel];
                    
                    return (
                      <button 
                        key={region.name}
                        className={`discovery-item ${selectedRegion === region.name ? 'active' : ''}`}
                        onClick={() => onSelectRegion(region.name)}
                      >
                        <div className="discovery-item-icon" style={{ backgroundColor: `${color}15`, color }}>
                          <AlertTriangleIcon />
                        </div>
                        <div className="discovery-item-info">
                          <span className="discovery-name">{region.name}</span>
                          <span className="discovery-zone">{ZONE_THAI[region.zone] ?? region.zone}</span>
                        </div>
                        <div className="discovery-item-temp" style={{ color }}>
                          {formatTemperature(temp)}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Toggle Discovery Button (visible on mobile or when closed) */}
        <button 
          className={`discovery-toggle ${showDiscovery ? 'active' : ''}`}
          onClick={() => setShowDiscovery(!showDiscovery)}
          title="แสดงพื้นที่เฝ้าระวัง"
        >
          <AlertTriangleIcon />
        </button>

        <div className="map-leaflet-container">
          <MapContainer
            center={[13.5, 101.5]}
            zoom={6}
            className="leaflet-map"
            zoomControl={false}
            scrollWheelZoom
          >
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            />

            <MapFlyTo region={activeRegionData} />

            {regions.map((region) => {
              const temperature = toSafeNumber(region.temperature);
              const risk = normalizeRisk(region.heatLevel, temperature);
              const color = HEAT_HEX[risk.heatLevel];
              const isSelected = selectedRegion === region.name;

              return (
                <CircleMarker
                  key={region.name}
                  center={[region.lat, region.lng]}
                  radius={isSelected ? 16 : 8}
                  pathOptions={{
                    color: isSelected ? '#ffffff' : color,
                    fillColor: color,
                    fillOpacity: isSelected ? 0.95 : 0.75,
                    weight: isSelected ? 2 : 1,
                    opacity: 1,
                  }}
                  eventHandlers={{
                    click: (event) => {
                      event.originalEvent.stopPropagation();
                      onSelectRegion(region.name);
                    },
                  }}
                >
                  <Tooltip
                    direction="top"
                    offset={[0, -10]}
                    opacity={1}
                    className="heat-tooltip"
                    permanent={isSelected}
                  >
                    <span className="tooltip-name">{region.name}</span>
                    <span className="tooltip-temp">{formatTemperature(temperature)}</span>
                  </Tooltip>
                </CircleMarker>
              );
            })}
          </MapContainer>
          
          {/* Map Legend Overlay */}
          <div className="map-legend-overlay">
            <h4 className="legend-title">ระดับความร้อน</h4>
            <div className="legend-items">
              {['normal', 'caution', 'warning', 'danger', 'severe'].map((level) => {
                const l = level as HeatLevel;
                return (
                  <div key={l} className="legend-row">
                    <span className="legend-color" style={{ backgroundColor: HEAT_HEX[l] }} />
                    <span className="legend-text">
                      {l === 'normal' && 'ปกติ'}
                      {l === 'caution' && 'เฝ้าระวัง'}
                      {l === 'warning' && 'เตือนภัย'}
                      {l === 'danger' && 'อันตราย'}
                      {l === 'severe' && 'รุนแรงมาก'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {activeRegionData && (
            <motion.div
              key="details-sidebar"
              className="map-sidebar"
              initial={{ x: '100%', opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: '100%', opacity: 0 }}
              transition={{ type: 'spring', damping: 26, stiffness: 220 }}
            >
              <div className="sidebar-header">
                <div className="sidebar-header-text">
                  <h2 className="sidebar-title">{activeRegionData.name}</h2>
                  <p className="sidebar-subtitle">
                    <LocationIcon /> {ZONE_THAI[activeRegionData.zone] ?? activeRegionData.zone}
                  </p>
                </div>
                <button className="close-button" onClick={() => onSelectRegion(null)} aria-label="ปิด">
                  <CloseIcon />
                </button>
              </div>

              <div className="sidebar-body">
                <div className="temperature-hero" style={{ 
                  background: `linear-gradient(135deg, ${HEAT_HEX[activeHeatLevel]}15, ${HEAT_HEX[activeHeatLevel]}05)` 
                }}>
                  <div className="temperature-label">อุณหภูมิคาดการณ์</div>
                  <span className="temperature-value" style={{ color: HEAT_HEX[activeHeatLevel] }}>
                    {formatTemperature(toSafeNumber(activeRegionData.temperature))}
                  </span>
                </div>

                <div className="badge-row">
                  <div
                    className="risk-badge"
                    style={{
                      backgroundColor: `${HEAT_HEX[activeHeatLevel]}15`,
                      color: HEAT_HEX[activeHeatLevel],
                      border: `1px solid ${HEAT_HEX[activeHeatLevel]}30`,
                    }}
                  >
                    <span className="risk-dot" style={{ backgroundColor: HEAT_HEX[activeHeatLevel], boxShadow: `0 0 6px ${HEAT_HEX[activeHeatLevel]}` }} />
                    ความเสี่ยง: {activeRegionRisk?.labelTh ?? 'ยังไม่มีข้อมูล'}
                  </div>
                </div>

                <div className="info-section">
                  <h3 className="info-heading">คำแนะนำด้านสุขภาพ</h3>
                  <div className="info-card advisory-card" style={{ borderLeftColor: HEAT_HEX[activeHeatLevel] }}>
                    <p className="info-paragraph">{HEAT_ADVICE_TEXT[activeHeatLevel]}</p>
                  </div>
                </div>

                <div className="info-section">
                  <h3 className="info-heading">พิกัดภูมิศาสตร์</h3>
                  <div className="coordinates-grid">
                    <div className="coordinate-card info-card">
                      <span className="coordinate-label">ละติจูด</span>
                      <span className="coordinate-value">{toSafeNumber(activeRegionData.lat).toFixed(4)}° N</span>
                    </div>
                    <div className="coordinate-card info-card">
                      <span className="coordinate-label">ลองจิจูด</span>
                      <span className="coordinate-value">{toSafeNumber(activeRegionData.lng).toFixed(4)}° E</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default MapComponent;
