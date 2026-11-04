// Server URL handling. Release builds talk HTTPS only; plain http is allowed for loopback hosts so
// the app can be tested against a local backend (emulator: 10.0.2.2, or `adb reverse` + 127.0.0.1).

const LOOPBACK = new Set(['localhost', '127.0.0.1', '10.0.2.2']);

export class ServerUrlError extends Error {}

export function normalizeServerUrl(input: string): string {
  let s = (input ?? '').trim();
  if (!s) throw new ServerUrlError('Enter the server address.');
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) s = `https://${s}`;
  let u: URL;
  try {
    u = new URL(s);
  } catch {
    throw new ServerUrlError('That is not a valid address.');
  }
  const proto = u.protocol.toLowerCase();
  const host = u.hostname.toLowerCase();
  if (proto !== 'https:' && !(proto === 'http:' && LOOPBACK.has(host))) {
    throw new ServerUrlError('Use an https:// address (for example your ngrok URL).');
  }
  if (u.username || u.password) throw new ServerUrlError('Do not put credentials in the address.');
  const path = u.pathname.replace(/\/+$/, '');
  return `${proto}//${u.host}${path}`;
}

/** Resolve a server-relative path (e.g. a signed media link) against the base URL. */
export function absUrl(base: string, path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${base.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
}

/** ws(s):// URL for a server-relative path. */
export function wsUrl(base: string, path: string): string {
  return absUrl(base, path).replace(/^http/i, 'ws');
}

/** scheme://host[:port] of an absolute URL. */
export function originOf(url: string): string {
  const m = /^(https?:\/\/[^/?#]+)/i.exec(url);
  return m ? m[1] : 'about:blank';
}
