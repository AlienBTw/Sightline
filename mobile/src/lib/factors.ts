// Plain-language names for the 11 hazard factors (training/configs/factors.yaml).
// F3/F4 have no usable training labels -> hidden in the app. F2/F5 are scored but muted (no alerts).
// Only F1 held up in live testing; every other factor is labelled "experimental" in the UI (display only,
// the backend thresholds are unchanged).

export interface FactorInfo {
  id: string;
  name: string;
  plain: string;
  hidden?: boolean;
  muted?: boolean;
}

/** Factors whose live detections were verified as reliable. */
export const RELIABLE_FACTORS: readonly string[] = ['F1'];

export function isExperimental(id: string | null | undefined): boolean {
  return !!id && !RELIABLE_FACTORS.includes(id);
}

export const EXPERIMENTAL_NOTE =
  'Experimental: only hand-in-danger-zone detections were reliable in live testing. Treat this as a hint and check the clip.';

export const FACTORS: FactorInfo[] = [
  { id: 'F1', name: 'Hand in danger zone', plain: "A hand or arm is reaching into a machine's danger zone." },
  { id: 'F2', name: 'No head protection', plain: 'Someone is not wearing a hard hat where one is needed.', muted: true },
  { id: 'F3', name: 'No eye/face protection', plain: 'No safety glasses or face shield near a machine.', hidden: true },
  { id: 'F4', name: 'Wrong glove state', plain: 'Gloves are wrong for the task near machinery.', hidden: true },
  { id: 'F5', name: 'No hi-vis vest', plain: 'Someone is not wearing hi-vis near vehicle traffic.', muted: true },
  { id: 'F6', name: 'Fall or slip', plain: 'A person is falling, slipping or lying on the floor.' },
  { id: 'F7', name: 'Unstable load', plain: 'A carried or stacked load looks unstable or overloaded.' },
  { id: 'F8', name: 'Person in vehicle path', plain: 'A person is off the walkway or in a forklift path.' },
  { id: 'F9', name: 'Open guard or panel', plain: 'A machine guard or electrical panel is open.' },
  { id: 'F10', name: 'Fire, smoke or sparks', plain: 'Visible fire, smoke or a spark shower.' },
  { id: 'F11', name: 'Abnormal machine behaviour', plain: 'A machine is moving or vibrating unexpectedly.' },
];

const BY_ID: Record<string, FactorInfo> = Object.fromEntries(FACTORS.map((f) => [f.id, f]));

export function factorInfo(id: string): FactorInfo {
  return BY_ID[id] ?? { id, name: id, plain: '' };
}

/** Factors shown in chips and filters (F3/F4 hidden). */
export const VISIBLE_FACTORS = FACTORS.filter((f) => !f.hidden);

export function visibleFactors(ids: string[] | undefined | null): string[] {
  return (ids ?? []).filter((id) => !BY_ID[id]?.hidden);
}

/** True if the event should raise an alert (it has at least one factor that is neither hidden nor muted). */
export function isAlerting(ids: string[] | undefined | null): boolean {
  return (ids ?? []).some((id) => !BY_ID[id]?.hidden && !BY_ID[id]?.muted);
}

/** Pick the factor to headline an event with, preferring the server's top factor if it is visible. */
export function headlineFactor(top: string | null | undefined, ids: string[]): string | null {
  // A reliable factor always wins the headline over an experimental one.
  const reliable = (ids ?? []).find((id) => RELIABLE_FACTORS.includes(id));
  if (reliable) return reliable;
  if (top && !BY_ID[top]?.hidden) return top;
  const vis = visibleFactors(ids);
  return vis.find((id) => !BY_ID[id]?.muted) ?? vis[0] ?? null;
}

export type RiskLevel = 'high' | 'medium' | 'low' | 'none';

export function riskLevel(risk: number | null | undefined): RiskLevel {
  if (risk == null || Number.isNaN(risk)) return 'none';
  if (risk >= 0.7) return 'high';
  if (risk >= 0.4) return 'medium';
  return 'low';
}
