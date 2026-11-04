// Shapes returned by the Sightline backend (backend/routes/*.py). Only the fields the app uses.

export type Role = 'admin' | 'demo' | string;

export interface SessionUser {
  username: string;
  role: Role;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  refresh_expires_in?: number;
  user: SessionUser;
}

export type CameraState =
  | 'stopped'
  | 'starting'
  | 'loading_model'
  | 'running'
  | 'reconnecting'
  | 'error'
  | 'finished'
  | string;

export interface CameraStatus {
  state: CameraState;
  device?: string | null;
  fps?: number | null;
  frames?: number;
  last_frame_age_s?: number | null;
  events?: number;
  last_error?: string | null;
  active_factors: string[];
  finished_reason?: string | null;
  uptime_s?: number;
}

export interface Camera {
  id: string;
  name: string;
  kind: 'virtual' | 'rtsp' | 'stream' | string;
  loop: boolean;
  enabled: boolean;
  max_runtime_s?: number | null;
  created_at?: number | string;
  video?: { duration_s?: number; width?: number; height?: number; fps?: number } | null;
  status: CameraStatus;
  preview: { snapshot_url: string; mjpeg_url: string; expires_in: number };
  source?: string;
  started?: boolean;
  start_error?: string | null;
}

export interface CameraList {
  items: Camera[];
  capacity: { active: number; max: number; gpus: string[] };
}

export type EventStatus = 'open' | 'acknowledged' | 'dismissed' | 'fix_pending';

export interface EventSummary {
  event_id: string;
  camera_id: string;
  started_at: string;
  ended_at?: string | null;
  risk_max?: number | null;
  top_factor?: string | null;
  top_factor_label?: string | null;
  factors: string[];
  factor_labels: Record<string, string>;
  status: EventStatus;
  escalated: boolean;
  thumb_url?: string;
}

export interface EventList {
  items: EventSummary[];
  total: number;
  limit: number;
  offset: number;
  next_offset: number | null;
}

export interface NormCircle {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  angle_deg?: number;
}

export interface Overlay {
  t_ms: number;
  factor: string;
  score?: number;
  risk?: number;
  circle?: { norm?: NormCircle; source?: string } | null;
  ellipse?: NormCircle | null; // older event schema
}

export interface Segment {
  factor: string;
  label?: string;
  start_s: number;
  end_s: number;
  peak_score?: number;
  threshold?: number;
}

export interface MuseVerdict {
  status: string;
  confirmed?: boolean | null;
  severity?: number | null;
  reason?: string | null;
  action?: string | null;
  error?: string;
}

export interface EscalationCall {
  factor: string;
  score?: number;
  threshold?: number;
  reason?: string;
  'sam3.1'?: { status: string; [k: string]: unknown };
  muse_spark?: MuseVerdict;
  final_circle?: { source?: string } | null;
}

export interface MediaRef {
  url: string;
  ready: boolean;
}

export interface EventDetail extends EventSummary {
  segments?: Segment[];
  overlays?: Overlay[];
  escalation?: {
    backends?: Record<string, unknown>;
    calls?: EscalationCall[];
  } | null;
  clip_size?: [number, number];
  clip_fps?: number;
  media: { clip: MediaRef; clip_circles: MediaRef; thumb: MediaRef; expires_in: number };
  actions: { status: string; by: string; at: number }[];
  model?: Record<string, unknown>;
}

export type WsMessage =
  | { type: 'hello'; user: string; cameras: { camera_id: string; state: string; active_factors: string[] }[] }
  | { type: 'event'; event: EventSummary }
  | { type: 'event_status'; event_id: string; status: EventStatus; by?: string }
  | { type: 'camera'; camera_id: string; state: string; error?: string | null }
  | { type: 'alert_state'; camera_id: string; factors: string[]; incident?: unknown }
  | { type: 'camera_added' | 'camera_removed'; camera_id: string }
  | { type: 'ping' | 'pong' };
