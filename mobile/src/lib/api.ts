// Sightline API client. Pure TypeScript (no React Native imports) so it can be unit-tested in Node
// and exercised against the real backend from a script.
//
// Security model:
// - The access token (15 min) lives only in memory; the refresh token lives in the TokenStore
//   (expo-secure-store in the app). Tokens never go into URLs: media uses the backend's signed
//   ?st= links and the WebSocket uses a one-time ticket.
// - 401 -> one single-flight refresh (rotation: the old refresh token is dead after use), then the
//   request is retried once. If the refresh itself is rejected the session is cleared and
//   onSessionExpired() fires so the UI can go back to the login screen.
import type {
  Camera,
  CameraList,
  EventDetail,
  EventList,
  EventStatus,
  SessionUser,
  TokenPair,
} from './types';

export interface TokenStore {
  getRefreshToken(): Promise<string | null>;
  setRefreshToken(token: string | null): Promise<void>;
}

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface ApiClientOptions {
  baseUrl: string;
  store: TokenStore;
  fetchFn?: FetchLike;
  timeoutMs?: number;
  onSessionExpired?: (reason: string) => void;
  onUser?: (user: SessionUser | null) => void;
  now?: () => number;
}

export class ApiError extends Error {
  status: number;
  retryAfter?: number;
  constructor(status: number, message: string, retryAfter?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.retryAfter = retryAfter;
  }
  get isNetwork() {
    return this.status === 0;
  }
}

export interface RequestOptions {
  body?: unknown;
  query?: Record<string, string | number | boolean | null | undefined>;
  auth?: boolean;
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface EventQuery {
  camera_id?: string;
  status?: string;
  factor?: string;
  min_risk?: number;
  limit?: number;
  offset?: number;
}

const COMMON_HEADERS: Record<string, string> = {
  Accept: 'application/json',
  // ngrok free shows an HTML interstitial to browser-like clients; this header skips it.
  'ngrok-skip-browser-warning': '1',
};

export function detailMessage(status: number, body: unknown): string {
  const d = (body as { detail?: unknown } | null)?.detail;
  if (typeof d === 'string' && d) return d;
  if (Array.isArray(d) && d.length) {
    return d
      .map((e) => (typeof e === 'object' && e && 'msg' in e ? String((e as { msg: unknown }).msg) : String(e)))
      .join('; ');
  }
  if (status === 0) return 'Cannot reach the server. Check the address and your connection.';
  if (status === 401) return 'Your session has ended. Please sign in again.';
  if (status === 403) return 'Your account is not allowed to do that.';
  if (status === 404) return 'Not found.';
  if (status === 429) return 'Too many requests. Wait a moment and try again.';
  if (status >= 500) return `Server error (${status}). Try again shortly.`;
  return `Request failed (${status}).`;
}

export function buildQuery(q?: RequestOptions['query']): string {
  if (!q) return '';
  const parts = Object.entries(q)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

export class ApiClient {
  readonly baseUrl: string;
  private store: TokenStore;
  private fetchFn: FetchLike;
  private timeoutMs: number;
  private onSessionExpired?: (reason: string) => void;
  private onUser?: (user: SessionUser | null) => void;
  private now: () => number;
  private access: string | null = null;
  private accessExp = 0;
  private refreshing: Promise<string> | null = null;
  private closed = false;
  user: SessionUser | null = null;

  constructor(o: ApiClientOptions) {
    this.baseUrl = o.baseUrl.replace(/\/+$/, '');
    this.store = o.store;
    this.fetchFn = o.fetchFn ?? ((input, init) => fetch(input, init));
    this.timeoutMs = o.timeoutMs ?? 15000;
    this.onSessionExpired = o.onSessionExpired;
    this.onUser = o.onUser;
    this.now = o.now ?? (() => Date.now());
  }

  url(path: string): string {
    if (/^https?:\/\//i.test(path)) return path;
    return `${this.baseUrl}/${path.replace(/^\/+/, '')}`;
  }

  get hasAccessToken() {
    return !!this.access;
  }

  // ---------------------------------------------------------------- transport
  private async send(method: string, path: string, opts: RequestOptions, token: string | null): Promise<Response> {
    const headers: Record<string, string> = { ...COMMON_HEADERS };
    let body: string | undefined;
    if (opts.body !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(opts.body);
    }
    if (token) headers.Authorization = `Bearer ${token}`;
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? this.timeoutMs);
    const onAbort = () => ctrl.abort();
    opts.signal?.addEventListener('abort', onAbort);
    try {
      return await this.fetchFn(this.url(path) + buildQuery(opts.query), {
        method,
        headers,
        body,
        signal: ctrl.signal,
      });
    } catch (e) {
      if (opts.signal?.aborted) throw e;
      throw new ApiError(0, detailMessage(0, null));
    } finally {
      clearTimeout(timeout);
      opts.signal?.removeEventListener('abort', onAbort);
    }
  }

  private async parse<T>(res: Response): Promise<T> {
    const text = await res.text();
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    if (!res.ok) {
      const ra = Number(res.headers.get('retry-after'));
      throw new ApiError(res.status, detailMessage(res.status, data), Number.isFinite(ra) && ra > 0 ? ra : undefined);
    }
    return data as T;
  }

  /** Authenticated JSON request with refresh-on-401 and a single retry. */
  async request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
    if (opts.auth === false) return this.parse<T>(await this.send(method, path, opts, null));
    let token = await this.ensureAccess();
    let res = await this.send(method, path, opts, token);
    if (res.status === 401) {
      // Another call may already have refreshed while this one was in flight.
      token = this.access && this.access !== token ? this.access : await this.refresh();
      res = await this.send(method, path, opts, token);
      if (res.status === 401) {
        await this.expire('unauthorized');
      }
    }
    return this.parse<T>(res);
  }

  /** A valid access token, refreshing proactively when it is about to expire. */
  async ensureAccess(): Promise<string> {
    if (this.closed) throw new ApiError(401, 'Signed out.');
    if (this.access && this.now() < this.accessExp) return this.access;
    return this.refresh();
  }

  /** Headers for native transfers (uploads) that cannot go through request(). */
  async authHeaders(): Promise<Record<string, string>> {
    const token = await this.ensureAccess();
    return { ...COMMON_HEADERS, Authorization: `Bearer ${token}` };
  }

  private setPair(p: TokenPair) {
    this.access = p.access_token;
    this.accessExp = this.now() + Math.max(10, (p.expires_in ?? 900) - 60) * 1000;
    this.user = p.user;
    this.onUser?.(p.user);
  }

  /** Single-flight refresh: concurrent callers share one /refresh call (rotation forbids reuse). */
  refresh(): Promise<string> {
    if (this.refreshing) return this.refreshing;
    this.refreshing = (async () => {
      try {
        const rt = await this.store.getRefreshToken();
        if (!rt) {
          await this.expire('no_session');
          throw new ApiError(401, 'Please sign in.');
        }
        const res = await this.send('POST', '/api/auth/refresh', { body: { refresh_token: rt } }, null);
        if (res.status === 401 || res.status === 403) {
          await this.expire('refresh_rejected');
          throw new ApiError(401, 'Your session has ended. Please sign in again.');
        }
        const pair = await this.parse<TokenPair>(res);
        await this.store.setRefreshToken(pair.refresh_token);
        this.setPair(pair);
        return pair.access_token;
      } finally {
        this.refreshing = null;
      }
    })();
    return this.refreshing;
  }

  private async expire(reason: string) {
    const had = !!this.access || !!this.user;
    this.access = null;
    this.accessExp = 0;
    this.user = null;
    await this.store.setRefreshToken(null);
    this.onUser?.(null);
    if (had || reason !== 'no_session') this.onSessionExpired?.(reason);
  }

  // ---------------------------------------------------------------- auth
  async login(username: string, password: string): Promise<SessionUser> {
    this.closed = false;
    const res = await this.send('POST', '/api/auth/login', { body: { username, password } }, null);
    if (res.status === 401) throw new ApiError(401, 'Wrong username or password.');
    const pair = await this.parse<TokenPair>(res);
    await this.store.setRefreshToken(pair.refresh_token);
    this.setPair(pair);
    return pair.user;
  }

  /** Resume a stored session (app start). Returns null when there is none or it was revoked. */
  async restore(): Promise<SessionUser | null> {
    const rt = await this.store.getRefreshToken();
    if (!rt) return null;
    try {
      await this.refresh();
      return this.user;
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) return null;
      throw e; // network trouble: keep the stored session, let the UI offer a retry
    }
  }

  async logout(): Promise<void> {
    const rt = await this.store.getRefreshToken();
    this.closed = true;
    this.access = null;
    this.user = null;
    await this.store.setRefreshToken(null);
    if (rt) {
      try {
        await this.send('POST', '/api/auth/logout', { body: { refresh_token: rt }, timeoutMs: 5000 }, null);
      } catch {
        // best effort: the refresh token is already gone from the device
      }
    }
  }

  me() {
    return this.request<{ username: string; role: string; permissions: string[] }>('GET', '/api/auth/me');
  }

  // ---------------------------------------------------------------- health
  health() {
    return this.request<{ status: string; version: string; uptime_s: number }>('GET', '/health', {
      auth: false,
      timeoutMs: 8000,
    });
  }

  healthDetail() {
    return this.request<Record<string, unknown>>('GET', '/api/health');
  }

  // ---------------------------------------------------------------- cameras
  cameras() {
    return this.request<CameraList>('GET', '/api/cameras');
  }
  camera(id: string) {
    return this.request<Camera>('GET', `/api/cameras/${encodeURIComponent(id)}`);
  }
  startCamera(id: string) {
    return this.request<Camera>('POST', `/api/cameras/${encodeURIComponent(id)}/start`, { timeoutMs: 30000 });
  }
  stopCamera(id: string) {
    return this.request<Camera>('POST', `/api/cameras/${encodeURIComponent(id)}/stop`, { timeoutMs: 30000 });
  }
  addCamera(name: string, uri: string, start = true) {
    return this.request<Camera>('POST', '/api/cameras', { body: { name, uri, start }, timeoutMs: 30000 });
  }
  deleteCamera(id: string) {
    return this.request<{ ok: boolean }>('DELETE', `/api/cameras/${encodeURIComponent(id)}`, { timeoutMs: 30000 });
  }

  // ---------------------------------------------------------------- events
  events(q: EventQuery = {}, signal?: AbortSignal) {
    return this.request<EventList>('GET', '/api/events', { query: { ...q }, signal });
  }
  event(id: string) {
    return this.request<EventDetail>('GET', `/api/events/${encodeURIComponent(id)}`);
  }
  ack(id: string) {
    return this.request<{ event_id: string; status: EventStatus }>('POST', `/api/events/${encodeURIComponent(id)}/ack`);
  }
  dismiss(id: string) {
    return this.request<{ event_id: string; status: EventStatus }>(
      'POST',
      `/api/events/${encodeURIComponent(id)}/dismiss`
    );
  }
  setStatus(id: string, status: EventStatus) {
    return this.request<{ event_id: string; status: EventStatus }>(
      'POST',
      `/api/events/${encodeURIComponent(id)}/status`,
      { body: { status } }
    );
  }

  // ---------------------------------------------------------------- realtime + push
  wsTicket() {
    return this.request<{ ticket: string; expires_in: number; url: string }>('POST', '/api/ws/ticket');
  }
  registerDevice(token: string, platform = 'android') {
    return this.request<{ ok: boolean }>('POST', '/api/devices', { body: { token, platform } });
  }
  unregisterDevice(token: string) {
    return this.request<{ ok: boolean }>('DELETE', '/api/devices', { query: { token } });
  }
  devices() {
    return this.request<{ items: { token: string; platform: string; last_seen: number }[] }>('GET', '/api/devices');
  }
  testPush() {
    return this.request<{ sent: number; http: number[] }>('POST', '/api/devices/test');
  }
}
