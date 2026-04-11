import React, { useMemo, useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import './Forecast.css';
import {
  formatProbabilityPercent,
  formatTemperature,
  formatThaiDate,
  formatThaiDayName,
  normalizeForecastEntry,
  type NormalizedForecastDay,
} from '../shared/heatNormalization';

export interface ForecastDay {
  day: number;
  date: string;
  day_name: string;
  temperature: number;
  risk_level: string;
  probability: number | null;
}

export interface ForecastProps {
  forecast: ForecastDay[];
  regionName: string | null;
  availableRegions: string[];
  onSelectRegion: (regionName: string) => void;
}

const AlertIcon = ({ className = '' }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const CalendarIcon = ({ className = '' }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
    <line x1="16" y1="2" x2="16" y2="6" />
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="10" x2="21" y2="10" />
  </svg>
);

const ChevronLeftIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="15 18 9 12 15 6" />
  </svg>
);

const ChevronRightIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="9 18 15 12 9 6" />
  </svg>
);

export const Forecast: React.FC<ForecastProps> = ({
  forecast,
  regionName,
  availableRegions,
  onSelectRegion,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const normalizedForecast = useMemo(
    () =>
      (forecast || [])
        .map((entry, index) => normalizeForecastEntry(entry, index + 1))
        .filter((entry): entry is NormalizedForecastDay => entry !== null),
    [forecast]
  );

  const maxTemp = useMemo(() => {
    if (normalizedForecast.length === 0) return 0;
    return Math.max(...normalizedForecast.map((d) => d.temperature));
  }, [normalizedForecast]);

  const minTemp = useMemo(() => {
    if (normalizedForecast.length === 0) return 0;
    return Math.min(...normalizedForecast.map((d) => d.temperature));
  }, [normalizedForecast]);

  const hottestDay = useMemo(() => {
    if (normalizedForecast.length === 0) return null;
    return normalizedForecast.reduce((prev, current) => 
      (prev.temperature > current.temperature) ? prev : current
    );
  }, [normalizedForecast]);

  const highRiskDaysCount = useMemo(() => {
    return normalizedForecast.filter(d => 
      d.risk.heatLevel === 'danger' || d.risk.heatLevel === 'severe'
    ).length;
  }, [normalizedForecast]);

  // Check scroll position
  const checkScroll = React.useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    
    const { scrollLeft, scrollWidth, clientWidth } = el;
    setCanScrollLeft(scrollLeft > 10);
    setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 10);
  }, []);

  useEffect(() => {
    checkScroll();
    const el = scrollRef.current;
    if (el) {
      el.addEventListener('scroll', checkScroll);
      window.addEventListener('resize', checkScroll);
      return () => {
        el.removeEventListener('scroll', checkScroll);
        window.removeEventListener('resize', checkScroll);
      };
    }
  }, [checkScroll, normalizedForecast]);

  const scroll = (direction: 'left' | 'right') => {
    const el = scrollRef.current;
    if (!el) return;
    
    const cardWidth = 140; // approx card width + gap
    const scrollAmount = direction === 'left' ? -cardWidth * 2 : cardWidth * 2;
    el.scrollBy({ left: scrollAmount, behavior: 'smooth' });
  };

  if (normalizedForecast.length === 0) {
    return (
      <div className="forecast-container empty">
        <p className="forecast-empty-text">ยังไม่มีข้อมูลพยากรณ์ (No forecast data available)</p>
      </div>
    );
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.08 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: 'spring' as const, stiffness: 260, damping: 20 } }
  };

  return (
    <div className="forecast-container">
      <div className="forecast-header-section">
        <div className="forecast-header-info">
          <h2 className="forecast-title">พยากรณ์ 7 วัน (7-Day Forecast)</h2>
          <p className="forecast-subtitle">
            {regionName ? `พื้นที่ที่เลือก: ${regionName} · ` : ''}
            อุณหภูมิคาดการณ์ช่วง {minTemp.toFixed(1)}–{maxTemp.toFixed(1)}°C
          </p>
        </div>

        {availableRegions.length > 0 && (
          <label className="forecast-region-picker">
            <span className="forecast-region-label">เลือกพื้นที่</span>
            <select
              className="forecast-region-select"
              value={regionName ?? availableRegions[0]}
              onChange={(event) => onSelectRegion(event.target.value)}
            >
              {availableRegions.map((region) => (
                <option key={region} value={region}>
                  {region}
                </option>
              ))}
            </select>
          </label>
        )}
        
        <div className="forecast-insights">
          <div className="insight-card">
            <div className="insight-icon-wrapper highlight">
              <AlertIcon />
            </div>
            <div className="insight-text">
              <span className="insight-label">จำนวนวันเสี่ยงสูง (High Risk Days)</span>
              <strong className="insight-value">{highRiskDaysCount} วัน</strong>
            </div>
          </div>
          
          {hottestDay && (
            <div className="insight-card">
              <div className="insight-icon-wrapper" style={{ color: `var(--heat-${hottestDay.risk.heatLevel})` }}>
                <CalendarIcon />
              </div>
              <div className="insight-text">
                <span className="insight-label">วันที่ร้อนที่สุด (Hottest Day)</span>
                <strong className="insight-value">
                  {formatThaiDayName(hottestDay.dayName, hottestDay.date)} - {formatTemperature(hottestDay.temperature)}
                </strong>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="forecast-content">
        <div className="forecast-scroll-wrapper">
          {canScrollLeft && (
            <button 
              className="forecast-scroll-btn forecast-scroll-btn--left" 
              onClick={() => scroll('left')}
              aria-label="เลื่อนไปทางซ้าย"
            >
              <ChevronLeftIcon />
            </button>
          )}
          
          <motion.div 
            ref={scrollRef}
            className="forecast-grid"
            variants={containerVariants}
            initial="hidden"
            animate="show"
          >
            {normalizedForecast.map((day, index) => {
              const riskInfo = day.risk;
              const dayLabel = formatThaiDayName(day.dayName, day.date) || `วันที่ ${day.day}`;
              const dateLabel = formatThaiDate(day.date);
              const probability = day.probability;
              
              let trend = 0;
              if (index > 0) {
                trend = day.temperature - normalizedForecast[index - 1].temperature;
              }

              const tempRange = Math.max(maxTemp - minTemp, 5);
              const heightPercent = 30 + ((day.temperature - minTemp) / tempRange) * 70;

              const cardStyle = {
                '--card-bg': `var(--heat-${riskInfo.heatLevel}-bg)`,
                '--card-border': `var(--heat-${riskInfo.heatLevel}-border)`,
                '--card-color': `var(--heat-${riskInfo.heatLevel})`,
                '--card-text': `var(--heat-${riskInfo.heatLevel}-text)`
              } as React.CSSProperties;

              const isToday = index === 0;

              return (
                <motion.div 
                  key={`day-${index}`} 
                  className={`forecast-card ${isToday ? 'forecast-card--today' : ''}`}
                  style={cardStyle}
                  variants={itemVariants}
                >
                  {isToday && <div className="fc-today-badge">วันนี้ (Today)</div>}
                  
                  <div className="fc-head">
                    <div className="fc-day-label">{dayLabel}</div>
                    <div className="fc-date-label">{dateLabel}</div>
                  </div>

                  <div className="fc-temp-wrapper">
                    <div className="fc-temp">
                      {formatTemperature(day.temperature)}
                    </div>
                    
                    <div className="fc-trend">
                      {index > 0 && trend !== 0 ? (
                        <span className={`trend-indicator ${trend > 0 ? 'trend-up' : 'trend-down'}`}>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                            {trend > 0 
                              ? <path d="M12 19V5M5 12l7-7 7 7"/> 
                              : <path d="M12 5v14M19 12l-7 7-7-7"/>}
                          </svg>
                          {Math.abs(trend).toFixed(1)}°
                        </span>
                      ) : (
                        <span className="trend-placeholder">━</span>
                      )}
                    </div>
                  </div>

                  <div className="fc-chart">
                    <div className="fc-chart-track">
                      <motion.div 
                        className="fc-chart-fill"
                        initial={{ height: 0 }}
                        animate={{ height: `${heightPercent}%` }}
                        transition={{ type: 'spring' as const, delay: index * 0.08 + 0.2, stiffness: 100 }}
                      />
                    </div>
                  </div>

                  <div className="fc-risk-badge">
                    {riskInfo.labelTh}
                  </div>

                  <div className="fc-probability">
                    <div className="fc-prob-label">โอกาสแม่นยำ {formatProbabilityPercent(probability)}</div>
                    <div className="fc-prob-track">
                      <motion.div 
                        className="fc-prob-fill"
                        initial={{ width: 0 }}
                        animate={{ width: probability === null ? 0 : `${Math.round(probability * 100)}%` }}
                        transition={{ ease: 'easeOut', delay: index * 0.08 + 0.4 }}
                      />
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
          
          {canScrollRight && (
            <button 
              className="forecast-scroll-btn forecast-scroll-btn--right" 
              onClick={() => scroll('right')}
              aria-label="เลื่อนไปทางขวา"
            >
              <ChevronRightIcon />
            </button>
          )}
        </div>
        
        {normalizedForecast.length > 3 && (
          <div className="forecast-scroll-hint">
            <span>← เลื่อนเพื่อดูวันเพิ่มเติม →</span>
          </div>
        )}
      </div>
    </div>
  );
};