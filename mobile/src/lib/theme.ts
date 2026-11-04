// Light industrial ops palette. Matches app/styles/sightline.css.
import type { RiskLevel } from './factors';

export const C = {
  bg: '#f4f6f8',
  surface: '#ffffff',
  surface2: '#eef2f6',
  border: '#d8dee6',
  text: '#1a2332',
  muted: '#5c6b7a',
  faint: '#7b8794',
  accent: '#c45c26',
  accentSoft: '#f3e6de',
  accentText: '#ffffff',
  danger: '#b42318',
  dangerSoft: '#fceceb',
  warn: '#b45309',
  warnSoft: '#fef3e2',
  low: '#a16207',
  ok: '#0f766e',
  okSoft: '#e6f2f1',
  info: '#1d4e89',
  infoSoft: '#e8eef5',
  onSolid: '#ffffff',
  overlay: 'rgba(26,35,50,0.55)',
};

export const FONT = {
  sans: 'IBMPlexSans_400Regular',
  sansMed: 'IBMPlexSans_500Medium',
  sansSem: 'IBMPlexSans_600SemiBold',
  mono: 'IBMPlexMono_500Medium',
};

export const RISK_COLOR: Record<RiskLevel, string> = {
  high: C.danger,
  medium: C.warn,
  low: C.low,
  none: C.faint,
};

export const STATUS_COLOR: Record<string, string> = {
  open: C.danger,
  acknowledged: C.info,
  fix_pending: C.warn,
  dismissed: C.faint,
};

export const STATUS_LABEL: Record<string, string> = {
  open: 'Open',
  acknowledged: 'Acknowledged',
  fix_pending: 'Fix pending',
  dismissed: 'Dismissed',
};

export const CAMERA_STATE: Record<string, { label: string; color: string }> = {
  running: { label: 'Live', color: C.ok },
  starting: { label: 'Starting', color: C.warn },
  loading_model: { label: 'Loading model', color: C.warn },
  reconnecting: { label: 'Reconnecting', color: C.warn },
  error: { label: 'Error', color: C.danger },
  finished: { label: 'Finished', color: C.faint },
  stopped: { label: 'Stopped', color: C.faint },
};

export const S = { xs: 4, sm: 8, md: 16, lg: 24, xl: 40 };
export const R = { sm: 6, md: 6, lg: 6 };
