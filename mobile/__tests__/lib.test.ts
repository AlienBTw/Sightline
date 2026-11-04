import { factorInfo, headlineFactor, isAlerting, isExperimental, riskLevel, verifyBadge, visibleFactors } from '../src/lib/factors';
import { absUrl, normalizeServerUrl, originOf, ServerUrlError, wsUrl } from '../src/lib/url';
test('server URLs: https only, except loopback for local testing', () => {
  expect(normalizeServerUrl('demo.ngrok-free.app')).toBe('https://demo.ngrok-free.app');
  expect(normalizeServerUrl(' https://demo.ngrok-free.app/ ')).toBe('https://demo.ngrok-free.app');
  expect(normalizeServerUrl('http://10.0.2.2:8000')).toBe('http://10.0.2.2:8000');
  expect(normalizeServerUrl('http://127.0.0.1:8000/')).toBe('http://127.0.0.1:8000');
  expect(() => normalizeServerUrl('http://demo.ngrok-free.app')).toThrow(ServerUrlError);
  expect(() => normalizeServerUrl('https://user:pw@demo.ngrok-free.app')).toThrow(ServerUrlError);
  expect(() => normalizeServerUrl('')).toThrow(ServerUrlError);
});

test('absUrl / wsUrl', () => {
  expect(absUrl('https://a.b', '/api/events/e/thumb.jpg?st=x')).toBe('https://a.b/api/events/e/thumb.jpg?st=x');
  expect(wsUrl('https://a.b', '/api/ws?ticket=t')).toBe('wss://a.b/api/ws?ticket=t');
  expect(wsUrl('http://127.0.0.1:8000', '/api/ws?ticket=t')).toBe('ws://127.0.0.1:8000/api/ws?ticket=t');
});

test('factor visibility and alerting rules (F3/F4 hidden, F2/F5 gated)', () => {
  expect(visibleFactors(['F1', 'F3', 'F4', 'F5'])).toEqual(['F1', 'F5']);
  expect(factorInfo('F2').name).toBe('No head protection');
  expect(factorInfo('F2').color).toBe('#c45c26');
  expect(factorInfo('F5').name).toBe('No hi-vis vest');
  expect(factorInfo('F5').color).toBe('#0f766e');
  expect(isAlerting(['F2', 'F5'])).toBe(false);
  expect(isAlerting(['F3'])).toBe(false);
  expect(isAlerting(['F5', 'F9'])).toBe(true);
  expect(isAlerting(['F5'])).toBe(false);
  expect(isAlerting(['F2'])).toBe(false);
  expect(isAlerting(['F5'], 'Muse Spark')).toBe(true);
  expect(isAlerting(['F2'], 'Muse Spark')).toBe(true);
  expect(headlineFactor('F2', ['F2'])).toBe('F2');
  expect(headlineFactor('F5', ['F5'])).toBe('F5');
  expect(headlineFactor('F3', ['F3', 'F5', 'F9'])).toBe('F9');
  expect(headlineFactor('F7', ['F7'])).toBe('F7');
});

test('verify-gate badges', () => {
  expect(verifyBadge({ status: 'confirmed', votes: 2, n_frames: 3 })).toBe('Verified by Muse Spark (2/3 frames)');
  expect(verifyBadge({ status: 'rejected' })).toBe('Not confirmed');
  expect(verifyBadge({ status: 'skipped_budget', reason: 'session_cap' })).toBe('Skipped: API limit');
  expect(verifyBadge({ status: 'failed' })).toBe('Check failed');
  expect(verifyBadge({ status: 'skipped_no_key' })).toBe('Check failed');
  expect(verifyBadge(null)).toBeNull();
});

test('risk levels', () => {
  expect(riskLevel(0.91)).toBe('high');
  expect(riskLevel(0.54)).toBe('medium');
  expect(riskLevel(0.2)).toBe('low');
  expect(riskLevel(null)).toBe('none');
});

describe('experimental factors', () => {
  it('only F1 is treated as reliable', () => {
    expect(isExperimental('F1')).toBe(false);
    expect(isExperimental('F9')).toBe(true);
    expect(isExperimental(null)).toBe(false);
  });
  it('headlines F1 over an experimental top factor', () => {
    expect(headlineFactor('F9', ['F9', 'F1'])).toBe('F1');
    expect(headlineFactor('F9', ['F9'])).toBe('F9');
  });
});

describe('live view origin', () => {
  it('extracts the server origin for the WebView base URL', () => {
    expect(originOf('https://x.ngrok-free.app/api/cameras/c/live.mjpeg?st=a')).toBe('https://x.ngrok-free.app');
    expect(originOf('http://127.0.0.1:8000/api/x')).toBe('http://127.0.0.1:8000');
    expect(originOf('nope')).toBe('about:blank');
  });
});
