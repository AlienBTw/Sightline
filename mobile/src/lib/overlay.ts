// Client-side circle overlay for clip.mp4, synced to playback time (overlays are timed by t_ms
// from the start of the clip, one model step per ~1 s).
import type { NormCircle, Overlay } from './types';

export interface ActiveCircle extends NormCircle {
  factor: string;
  score?: number;
  key: string;
}

export function overlayCircle(o: Overlay): NormCircle | null {
  const c = o.circle?.norm ?? o.ellipse ?? null;
  if (!c || ![c.cx, c.cy, c.rx, c.ry].every((v) => typeof v === 'number' && Number.isFinite(v))) return null;
  return c;
}

/**
 * Circles to draw at playback time `tMs`: for each factor, the latest overlay at or before tMs
 * (plus a small `leadMs` look-ahead for decoder jitter), held for `holdMs` after its timestamp.
 */
export function circlesAt(
  overlays: Overlay[] | undefined | null,
  tMs: number,
  opts: { holdMs?: number; leadMs?: number; hidden?: (factor: string) => boolean } = {}
): ActiveCircle[] {
  const holdMs = opts.holdMs ?? 1200;
  const leadMs = opts.leadMs ?? 150;
  if (!overlays?.length || !Number.isFinite(tMs)) return [];
  const best = new Map<string, Overlay>();
  for (const o of overlays) {
    if (opts.hidden?.(o.factor)) continue;
    if (o.t_ms > tMs + leadMs || tMs - o.t_ms > holdMs) continue;
    const prev = best.get(o.factor);
    if (!prev || o.t_ms > prev.t_ms) best.set(o.factor, o);
  }
  const out: ActiveCircle[] = [];
  for (const [factor, o] of best) {
    const c = overlayCircle(o);
    if (c) out.push({ ...c, factor, score: o.score, key: `${factor}@${o.t_ms}` });
  }
  return out.sort((a, b) => a.factor.localeCompare(b.factor, undefined, { numeric: true }));
}

/** Pixel box for a normalised circle inside a WxH video frame. */
export function circleBox(c: NormCircle, w: number, h: number) {
  const width = Math.max(8, 2 * c.rx * w);
  const height = Math.max(8, 2 * c.ry * h);
  return { left: c.cx * w - width / 2, top: c.cy * h - height / 2, width, height, rotate: `${c.angle_deg ?? 0}deg` };
}

/** Times (ms) of overlay steps, for timeline markers. */
export function overlayMarkers(overlays: Overlay[] | undefined | null): number[] {
  return [...new Set((overlays ?? []).map((o) => o.t_ms))].sort((a, b) => a - b);
}
