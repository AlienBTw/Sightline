import { ApiClient, ApiError, buildQuery, type TokenStore } from '../src/lib/api';

/** In-memory fake of the backend's auth: access tokens + rotating refresh tokens. */
function fakeServer(opts: { accessValid?: (t: string) => boolean } = {}) {
  let n = 0;
  const live = new Set<string>();
  const valid = new Set<string>();
  const calls: { method: string; url: string; auth?: string }[] = [];
  let refreshCalls = 0;
  const pair = () => {
    n += 1;
    const at = `at${n}`;
    const rt = `rt${n}_xxxxxxxxxxxxxxxxxxxx`;
    valid.add(at);
    live.add(rt);
    return { access_token: at, refresh_token: rt, token_type: 'bearer', expires_in: 900, user: { username: 'judge', role: 'demo' } };
  };
  const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
    new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });

  const fetchFn = async (url: string, init?: RequestInit) => {
    const h = (init?.headers ?? {}) as Record<string, string>;
    const method = init?.method ?? 'GET';
    calls.push({ method, url, auth: h.Authorization });
    const path = new URL(url).pathname;
    const body = init?.body ? JSON.parse(String(init.body)) : {};
    if (path === '/api/auth/login') {
      return body.password === 'pw' ? json(200, pair()) : json(401, { detail: 'invalid username or password' });
    }
    if (path === '/api/auth/refresh') {
      refreshCalls += 1;
      await new Promise((r) => setTimeout(r, 5));
      if (!live.has(body.refresh_token)) {
        live.clear(); // reuse -> whole family revoked
        return json(401, { detail: 'refresh token already used; please log in again' });
      }
      live.delete(body.refresh_token);
      return json(200, pair());
    }
    if (path === '/api/auth/logout') return json(200, { ok: true });
    const tok = (h.Authorization ?? '').replace('Bearer ', '');
    const ok = opts.accessValid ? opts.accessValid(tok) : valid.has(tok);
    if (!ok) return json(401, { detail: 'token expired' }, { 'www-authenticate': 'Bearer' });
    if (path === '/api/cameras') return json(200, { items: [], capacity: { active: 0, max: 6, gpus: [] } });
    if (path === '/api/events') return json(200, { items: [], total: 0, limit: 30, offset: 0, next_offset: null });
    return json(404, { detail: 'not found' });
  };
  return {
    fetchFn,
    calls,
    get refreshCalls() {
      return refreshCalls;
    },
    expireAllAccess: () => valid.clear(),
    revokeAll: () => live.clear(),
  };
}

function memStore(initial: string | null = null): TokenStore & { value: string | null } {
  const s = {
    value: initial,
    getRefreshToken: async () => s.value,
    setRefreshToken: async (t: string | null) => {
      s.value = t;
    },
  };
  return s;
}

test('login stores the refresh token and sends the bearer access token', async () => {
  const srv = fakeServer();
  const store = memStore();
  const api = new ApiClient({ baseUrl: 'https://x.ngrok-free.app/', store, fetchFn: srv.fetchFn });
  const user = await api.login('judge', 'pw');
  expect(user.role).toBe('demo');
  expect(store.value).toMatch(/^rt1_/);
  await api.cameras();
  const last = srv.calls[srv.calls.length - 1];
  expect(last.url).toBe('https://x.ngrok-free.app/api/cameras');
  expect(last.auth).toBe('Bearer at1');
});

test('wrong password gives a friendly 401 and stores nothing', async () => {
  const srv = fakeServer();
  const store = memStore();
  const api = new ApiClient({ baseUrl: 'https://x', store, fetchFn: srv.fetchFn });
  await expect(api.login('judge', 'nope')).rejects.toMatchObject({ status: 401, message: 'Wrong username or password.' });
  expect(store.value).toBeNull();
});

test('401 -> one refresh with rotation, then the request is retried', async () => {
  const srv = fakeServer();
  const store = memStore();
  const api = new ApiClient({ baseUrl: 'https://x', store, fetchFn: srv.fetchFn });
  await api.login('judge', 'pw');
  srv.expireAllAccess();
  const r = await api.events();
  expect(r.total).toBe(0);
  expect(srv.refreshCalls).toBe(1);
  expect(store.value).toMatch(/^rt2_/); // rotated
});

test('concurrent 401s share a single refresh (no refresh-token reuse)', async () => {
  const srv = fakeServer();
  const store = memStore();
  const expired = jest.fn();
  const api = new ApiClient({ baseUrl: 'https://x', store, fetchFn: srv.fetchFn, onSessionExpired: expired });
  await api.login('judge', 'pw');
  srv.expireAllAccess();
  await Promise.all([api.cameras(), api.events(), api.cameras(), api.events()]);
  expect(srv.refreshCalls).toBe(1);
  expect(expired).not.toHaveBeenCalled();
});

test('refresh rejected -> session cleared and onSessionExpired fires', async () => {
  const srv = fakeServer();
  const store = memStore();
  const expired = jest.fn();
  const users: unknown[] = [];
  const api = new ApiClient({ baseUrl: 'https://x', store, fetchFn: srv.fetchFn, onSessionExpired: expired, onUser: (u) => users.push(u) });
  await api.login('judge', 'pw');
  srv.expireAllAccess();
  srv.revokeAll();
  await expect(api.cameras()).rejects.toBeInstanceOf(ApiError);
  expect(expired).toHaveBeenCalledWith('refresh_rejected');
  expect(store.value).toBeNull();
  expect(users[users.length - 1]).toBeNull();
});

test('access token is refreshed proactively before it expires', async () => {
  const srv = fakeServer();
  let now = 1_000_000;
  const api = new ApiClient({ baseUrl: 'https://x', store: memStore(), fetchFn: srv.fetchFn, now: () => now });
  await api.login('judge', 'pw');
  now += 900_000; // 15 min later
  await api.cameras();
  expect(srv.refreshCalls).toBe(1);
  expect(srv.calls.filter((c) => c.url.endsWith('/api/cameras')).every((c) => c.auth === 'Bearer at2')).toBe(true);
});

test('restore(): network failure keeps the stored session; a revoked one returns null', async () => {
  const store = memStore('rt_stored_xxxxxxxxxxxxxxxx');
  const down = new ApiClient({
    baseUrl: 'https://x',
    store,
    fetchFn: async () => {
      throw new TypeError('Network request failed');
    },
  });
  await expect(down.restore()).rejects.toMatchObject({ status: 0 });
  expect(store.value).toBe('rt_stored_xxxxxxxxxxxxxxxx');

  const srv = fakeServer();
  const api = new ApiClient({ baseUrl: 'https://x', store, fetchFn: srv.fetchFn });
  expect(await api.restore()).toBeNull();
  expect(store.value).toBeNull();
});

test('tokens never appear in request URLs', async () => {
  const srv = fakeServer();
  const api = new ApiClient({ baseUrl: 'https://x', store: memStore(), fetchFn: srv.fetchFn });
  await api.login('judge', 'pw');
  srv.expireAllAccess();
  await api.events({ status: 'open', factor: 'F1', limit: 5 });
  await api.logout();
  for (const c of srv.calls) {
    expect(c.url).not.toMatch(/at\d|rt\d/);
  }
});

test('logout clears the stored token and blocks further calls', async () => {
  const srv = fakeServer();
  const store = memStore();
  const api = new ApiClient({ baseUrl: 'https://x', store, fetchFn: srv.fetchFn });
  await api.login('judge', 'pw');
  await api.logout();
  expect(store.value).toBeNull();
  await expect(api.cameras()).rejects.toMatchObject({ status: 401 });
});

test('buildQuery skips empty values and encodes the rest', () => {
  expect(buildQuery({ a: 'x y', b: undefined, c: '', d: 0, e: null })).toBe('?a=x%20y&d=0');
  expect(buildQuery({})).toBe('');
});
