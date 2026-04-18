import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Overview.css';
import {
  HEAT_ADVICE_TEXT,
  HEAT_LABEL_TH,
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
  probability?: number;
}

export interface OverviewProps {
  regions: Region[];
  selectedRegion: string | null;
  onSelectRegion: (name: string | null) => void;
}

const HEAT_COLORS: Record<HeatLevel, string> = {
  normal: 'var(--heat-normal)',
  caution: 'var(--heat-caution)',
  warning: 'var(--heat-warning)',
  danger: 'var(--heat-danger)',
  severe: 'var(--heat-severe)',
};

const HEAT_BG: Record<HeatLevel, string> = {
  normal: 'var(--heat-normal-bg)',
  caution: 'var(--heat-caution-bg)',
  warning: 'var(--heat-warning-bg)',
  danger: 'var(--heat-danger-bg)',
  severe: 'var(--heat-severe-bg)',
};

const HEAT_TEXT: Record<HeatLevel, string> = {
  normal: 'var(--heat-normal-text)',
  caution: 'var(--heat-caution-text)',
  warning: 'var(--heat-warning-text)',
  danger: 'var(--heat-danger-text)',
  severe: 'var(--heat-severe-text)',
};

const HEAT_ORDER: HeatLevel[] = ['normal', 'caution', 'warning', 'danger', 'severe'];

const toSafeNumber = (value: unknown, fallback = 0): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const ChevronIcon = ({ open }: { open: boolean }) => (
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
    style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}
  >
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const SearchIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </svg>
);

const AlertIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
    <line x1="12" y1="9" x2="12" y2="13"></line>
    <line x1="12" y1="17" x2="12.01" y2="17"></line>
  </svg>
);

export const Overview: React.FC<OverviewProps> = ({ regions, onSelectRegion }) => {
  const [expandedRegion, setExpandedRegion] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterZone, setFilterZone] = useState<string>('all');

  const stats = useMemo(() => {
    if (regions.length === 0) {
      return { alerts: 0, maxTemp: 0, minTemp: 0, avgTemp: 0, hottestRegion: null as Region | null, warningCount: 0, dangerCount: 0 };
    }

    let maxTemp = -Infinity;
    let minTemp = Infinity;
    let sumTemp = 0;
    let alerts = 0;
    let warningCount = 0;
    let dangerCount = 0;
    let hottestRegion: Region | null = null;

    regions.forEach((region) => {
      const temperature = toSafeNumber(region.temperature);
      const risk = normalizeRisk(region.heatLevel, temperature);
      if (temperature > maxTemp) {
        maxTemp = temperature;
        hottestRegion = region;
      }
      if (temperature < minTemp) minTemp = temperature;
      sumTemp += temperature;
      if (risk.heatLevel !== 'normal') alerts++;
      if (risk.heatLevel === 'warning') warningCount++;
      if (risk.heatLevel === 'danger' || risk.heatLevel === 'severe') dangerCount++;
    });

    return {
      alerts,
      maxTemp,
      minTemp,
      avgTemp: sumTemp / regions.length,
      hottestRegion,
      warningCount,
      dangerCount
    };
  }, [regions]);

  const uniqueZones = useMemo(() => {
    const zones = new Set(regions.map(r => r.zone));
    return Array.from(zones);
  }, [regions]);

  const alertRegions = useMemo(() => {
    return [...regions]
      .map((region) => {
        const temperature = toSafeNumber(region.temperature);
        const risk = normalizeRisk(region.heatLevel, temperature);
        return { region, temperature, risk };
      })
      .filter(({ risk }) => risk.heatLevel === 'warning' || risk.heatLevel === 'danger' || risk.heatLevel === 'severe')
      .sort((a, b) => toSafeNumber(b.temperature) - toSafeNumber(a.temperature));
  }, [regions]);

  const filteredAndSortedRegions = useMemo(() => {
    return [...regions]
      .filter(r => {
        const matchesSearch = r.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                              (ZONE_THAI[r.zone] || r.zone).includes(searchQuery);
        const matchesZone = filterZone === 'all' || r.zone === filterZone;
        return matchesSearch && matchesZone;
      })
      .sort((a, b) => toSafeNumber(b.temperature) - toSafeNumber(a.temperature));
  }, [regions, searchQuery, filterZone]);

  const handleRowClick = (name: string) => {
    const next = expandedRegion === name ? null : name;
    setExpandedRegion(next);
    onSelectRegion(next);
  };

  return (
    <div className="overview-container">
      <header className="overview-header">
        <div className="overview-title-group">
          <h1 className="overview-title">ภาพรวมคลื่นความร้อน</h1>
          <p className="overview-subtitle">รายงานสถานการณ์อุณหภูมิและความเสี่ยงทั่วประเทศ</p>
        </div>
      </header>

      {/* New Alerts Dashboard Section */}
      <div className="dashboard-summary">
        <div className="summary-card main-alert">
          <div className="card-icon-wrapper danger-bg">
            <AlertIcon />
          </div>
          <div className="card-content">
            <span className="card-label">เฝ้าระวังสูงสุด</span>
            <div className="card-value-group">
              <span className="card-value text-danger">{stats.hottestRegion ? stats.hottestRegion.name : '-'}</span>
              <span className="card-sub-value">{formatTemperature(stats.maxTemp)}</span>
            </div>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-content">
            <span className="card-label">พื้นที่เตือนภัย (Warning)</span>
            <span className="card-value text-warning">{stats.warningCount} <span className="unit">พื้นที่</span></span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-content">
            <span className="card-label">พื้นที่อันตราย (Danger)</span>
            <span className="card-value text-danger">{stats.dangerCount} <span className="unit">พื้นที่</span></span>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-content">
            <span className="card-label">อุณหภูมิเฉลี่ยประเทศ</span>
            <span className="card-value">{formatTemperature(stats.avgTemp)}</span>
          </div>
        </div>
      </div>

      <section className="alerts-panel" aria-labelledby="alerts-panel-title">
        <div className="alerts-panel-header">
          <div>
            <h2 id="alerts-panel-title" className="alerts-panel-title">การแจ้งเตือนพื้นที่เสี่ยง</h2>
            <p className="alerts-panel-subtitle">สรุปพื้นที่ที่ควรเฝ้าระวังเป็นพิเศษในขณะนี้</p>
          </div>
          <span className="alerts-panel-count">{alertRegions.length} รายการ</span>
        </div>

        {alertRegions.length === 0 ? (
          <div className="alerts-empty-state">ขณะนี้ยังไม่มีพื้นที่ที่อยู่ในระดับเตือนภัยขึ้นไป</div>
        ) : (
          <div className="alerts-list" role="list">
            {alertRegions.map(({ region, temperature, risk }) => {
              const color = HEAT_COLORS[risk.heatLevel];
              const bg = HEAT_BG[risk.heatLevel];
              const text = HEAT_TEXT[risk.heatLevel];

              return (
                <button
                  key={region.name}
                  type="button"
                  className="alert-item"
                  onClick={() => handleRowClick(region.name)}
                >
                  <div className="alert-item-leading" style={{ backgroundColor: bg, color: text }}>
                    <AlertIcon />
                  </div>
                  <div className="alert-item-body">
                    <div className="alert-item-topline">
                      <span className="alert-item-name">{region.name}</span>
                      <span className="alert-item-temp" style={{ color }}>{formatTemperature(temperature)}</span>
                    </div>
                    <div className="alert-item-meta">
                      <span>{ZONE_THAI[region.zone] ?? region.zone}</span>
                      <span className="alert-item-separator">•</span>
                      <span>ระดับ {risk.labelTh}</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </section>

      <div className="regions-section">
        <div className="section-header-controls">
          <h2 className="section-heading">สถานะรายพื้นที่</h2>
          
          <div className="controls-group">
            <div className="search-box">
              <SearchIcon />
              <input 
                type="text" 
                placeholder="ค้นหาจังหวัด..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <select 
              className="zone-filter"
              value={filterZone}
              onChange={(e) => setFilterZone(e.target.value)}
            >
              <option value="all">ทุกภูมิภาค</option>
              {uniqueZones.map(zone => (
                <option key={zone} value={zone}>{ZONE_THAI[zone] || zone}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="regions-list">
          {filteredAndSortedRegions.length === 0 ? (
            <div className="empty-state">ไม่พบข้อมูลที่ตรงกับการค้นหา</div>
          ) : (
            filteredAndSortedRegions.map((region, index) => {
              const temperature = toSafeNumber(region.temperature);
              const risk = normalizeRisk(region.heatLevel, temperature);
              const bg = HEAT_BG[risk.heatLevel];
              const color = HEAT_COLORS[risk.heatLevel];
              const text = HEAT_TEXT[risk.heatLevel];
              const isTop = index === 0 && filterZone === 'all' && !searchQuery;
              const isExpanded = expandedRegion === region.name;
              const barWidth = Math.max(0, Math.min(100, ((temperature - 28) / (44 - 28)) * 100));

              return (
                <div key={region.name} className="region-item">
                  <motion.div
                    onClick={() => handleRowClick(region.name)}
                    className={`region-row ${isTop ? 'region-row-primary' : ''} ${isExpanded ? 'expanded' : ''}`}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.35, delay: Math.min(index * 0.04, 0.5), ease: 'easeOut' }}
                  >
                    <div className="region-rank">{index + 1}</div>

                    <div className="region-name-col">
                      <div className="region-name">{region.name}</div>
                      <div className="region-zone">{ZONE_THAI[region.zone] ?? region.zone}</div>
                    </div>

                    <div className="region-bar-col">
                      <div className="temp-bar-track">
                        <div className="temp-bar-fill" style={{ width: `${barWidth}%`, backgroundColor: color }} />
                      </div>
                    </div>

                    <div className="region-temp-col">{formatTemperature(temperature)}</div>

                    <div className="region-badge-col">
                      <div className="region-badge" style={{ backgroundColor: bg, color: text }}>
                        {risk.labelTh}
                      </div>
                    </div>

                    <div className="region-chevron" style={{ color: 'var(--text-tertiary)' }}>
                      <ChevronIcon open={isExpanded} />
                    </div>
                  </motion.div>

                  <AnimatePresence initial={false}>
                    {isExpanded && (
                      <motion.div
                        key="detail"
                        className="region-detail-panel"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.28, ease: 'easeInOut' }}
                        style={{ overflow: 'hidden' }}
                      >
                        <div className="detail-inner">
                          <div className="detail-top">
                            <div className="detail-temp" style={{ color }}>
                              {formatTemperature(temperature)}
                            </div>
                            <div className="detail-badge" style={{ backgroundColor: bg, color: text }}>
                              ระดับ: {risk.labelTh}
                            </div>
                          </div>

                          <div className="detail-grid">
                            <div className="detail-cell">
                              <span className="detail-cell-label">ภาค</span>
                              <span className="detail-cell-value">{ZONE_THAI[region.zone] ?? region.zone}</span>
                            </div>
                            <div className="detail-cell">
                              <span className="detail-cell-label">ละติจูด</span>
                              <span className="detail-cell-value mono">{toSafeNumber(region.lat).toFixed(4)}</span>
                            </div>
                            <div className="detail-cell">
                              <span className="detail-cell-label">ลองจิจูด</span>
                              <span className="detail-cell-value mono">{toSafeNumber(region.lng).toFixed(4)}</span>
                            </div>
                            {region.probability !== undefined && (
                              <div className="detail-cell">
                                <span className="detail-cell-label">ความเชื่อมั่น (Confidence)</span>
                                <span className="detail-cell-value">{Math.round(region.probability * 100)}%</span>
                              </div>
                            )}
                          </div>

                          <div className="detail-alert-box" style={{ borderColor: color, backgroundColor: bg }}>
                            <div className="detail-alert-icon" style={{ color: text }}>
                              <AlertIcon />
                            </div>
                            <div className="detail-advice">
                              <span className="detail-advice-label" style={{ color: text }}>คำแนะนำการปฏิบัติตัว</span>
                              <p className="detail-advice-text">{HEAT_ADVICE_TEXT[risk.heatLevel]}</p>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="legend-ribbon">
        <span className="legend-label">ระดับความร้อน</span>
        <div className="legend-scale">
          {HEAT_ORDER.map((level) => (
            <div key={level} className="scale-item">
              <div className="scale-dot" style={{ backgroundColor: HEAT_COLORS[level] }} />
              <span className="scale-name">{HEAT_LABEL_TH[level]}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Overview;
