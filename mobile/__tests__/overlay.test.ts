import { circleBox, circlesAt, overlayMarkers } from '../src/lib/overlay';
import type { Overlay } from '../src/lib/types';

const ov = (t_ms: number, factor: string, cx = 0.5): Overlay => ({
  t_ms,
  factor,
  score: 0.2,
  circle: { norm: { cx, cy: 0.5, rx: 0.05, ry: 0.08, angle_deg: 0 } },
});

const overlays: Overlay[] = [ov(4000, 'F9', 0.1), ov(5000, 'F9', 0.2), ov(6000, 'F9', 0.3), ov(5000, 'F7', 0.7), ov(5000, 'F3')];

test('nothing before the first step', () => {
  expect(circlesAt(overlays, 3000)).toEqual([]);
});

test('picks the latest step at or before t for each factor', () => {
  const c = circlesAt(overlays, 5400);
  expect(c.map((x) => [x.factor, x.cx])).toEqual([
    ['F3', 0.5],
    ['F7', 0.7],
    ['F9', 0.2],
  ]);
});

test('small look-ahead absorbs decoder jitter', () => {
  expect(circlesAt(overlays, 3900).map((x) => x.cx)).toEqual([0.1]);
});

test('a step is held for holdMs then disappears', () => {
  expect(circlesAt(overlays, 7100).map((x) => x.factor)).toEqual(['F9']);
  expect(circlesAt(overlays, 7300)).toEqual([]);
});

test('hidden factors are filtered', () => {
  expect(circlesAt(overlays, 5000, { hidden: (f) => f === 'F3' }).map((x) => x.factor)).toEqual(['F7', 'F9']);
});

test('supports the older ellipse schema and skips malformed circles', () => {
  const legacy: Overlay[] = [
    { t_ms: 0, factor: 'F1', ellipse: { cx: 0.4, cy: 0.6, rx: 0.08, ry: 0.05, angle_deg: 30 } },
    { t_ms: 0, factor: 'F2', circle: { norm: { cx: NaN, cy: 0, rx: 0, ry: 0 } } },
  ];
  const c = circlesAt(legacy, 100);
  expect(c).toHaveLength(1);
  expect(c[0]).toMatchObject({ factor: 'F1', angle_deg: 30 });
});

test('circleBox maps normalised coords to pixels', () => {
  const b = circleBox({ cx: 0.5, cy: 0.8689, rx: 0.0459, ry: 0.0816 }, 960, 540);
  expect(b.width).toBeCloseTo(88.1, 1);
  expect(b.height).toBeCloseTo(88.1, 1);
  expect(b.left + b.width / 2).toBeCloseTo(480, 3);
  expect(b.top + b.height / 2).toBeCloseTo(469.2, 1);
});

test('markers are unique and sorted', () => {
  expect(overlayMarkers(overlays)).toEqual([4000, 5000, 6000]);
  expect(overlayMarkers(undefined)).toEqual([]);
});
