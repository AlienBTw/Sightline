// Exercise the app's real ApiClient (src/lib/api.ts) against a running backend, from Node 22+.
//   SL_BASE=http://127.0.0.1:8000 SL_USER=... SL_PASS=... \
//   node scripts/api-smoke.mts
// Credentials come from the environment only. Prints one line per check; exit code 1 on failure.
import { ApiClient, ApiError, type TokenStore } from '../src/lib/api.ts';

const BASE = process.env.SL_BASE ?? 'http://127.0.0.1:8000';
const USER = process.env.SL_USER ?? '';
const PASS = process.env.SL_PASS ?? '';

let failed = 0;
const results: string[] = [];
async function check(name: string, fn: () => Promise<string | void>) {
  const t0 = Date.now();
  try {
    const info = await fn();
    results.push(`PASS ${name}${info ? ` - ${info}` : ''} (${Date.now() - t0} ms)`);
  } catch (e) {
    failed += 1;
    results.push(`FAIL ${name} - ${e instanceof Error ? e.message : String(e)}`);
  }
  console.log(results[results.length - 1]);
}
function assert(c: unknown, msg: string): asserts c {
  if (!c) throw new Error(msg);
}
function memStore(): TokenStore & { value: string | null } {
  const s = { value: null as string | null, getRefreshToken: async () => s.value, setRefreshToken: async (t: string | null) => void (s.value = t) };
  return s;
}

const store = memStore();
let expired = '';
const api = new ApiClient({ baseUrl: BASE, store, onSessionExpired: (r) => (expired = r) });

await check('GET /health (public)', async () => {
  const h = await api.health();
  return `status=${h.status} version=${h.version}`;
});
await check('login with a wrong password -> 401', async () => {
  const bad = new ApiClient({ baseUrl: BASE, store: memStore() });
  try {
    await bad.login(USER, PASS + 'x');
    throw new Error('accepted a wrong password');
  } catch (e) {
    assert(e instanceof ApiError && e.status === 401, `expected 401, got ${String(e)}`);
  }
});
await check('login', async () => {
  const u = await api.login(USER, PASS);
  assert(store.value, 'no refresh token stored');
  return `user=${u.username} role=${u.role}`;
});
await check('GET /api/auth/me', async () => `perms=${(await api.me()).permissions.join(',')}`);

await check('refresh rotation + reuse detection', async () => {
  const old = store.value!;
  await api.refresh();
  assert(store.value && store.value !== old, 'refresh token not rotated');
  const res = await fetch(`${BASE}/api/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: old }),
  });
  assert(res.status === 401, `reused token gave ${res.status}`);
  // reuse revokes the whole family -> our current refresh token is dead too; log in again
  await api.login(USER, PASS);
  return 'rotated; reused token rejected (family revoked)';
});

await check('401 -> auto refresh -> retry (forced stale access token)', async () => {
  (api as unknown as { access: string }).access = 'stale.invalid.token';
  (api as unknown as { accessExp: number }).accessExp = Date.now() + 600_000;
  const r = await api.cameras();
  return `cameras=${r.items.length} after transparent refresh`;
});

await check('concurrent 401s share one refresh', async () => {
  (api as unknown as { access: string }).access = 'stale.invalid.token';
  (api as unknown as { accessExp: number }).accessExp = Date.now() + 600_000;
  await Promise.all([api.cameras(), api.events({ limit: 1 }), api.cameras(), api.events({ limit: 1 })]);
  assert(!expired, `session expired: ${expired}`);
  return 'no reuse, session alive';
});

let eventId = '';
await check('GET /api/events (feed, filters, pagination)', async () => {
  const r = await api.events({ limit: 3 });
  eventId = r.items[0]?.event_id ?? '';
  const open = await api.events({ status: 'open', limit: 1 });
  const f = await api.events({ factor: 'F9', limit: 1 });
  return `total=${r.total} next=${r.next_offset} open=${open.total} F9=${f.total} thumb=${!!r.items[0]?.thumb_url}`;
});

if (eventId) {
  await check('GET /api/events/{id} (overlays, escalation, signed media)', async () => {
    const d = await api.event(eventId);
    assert(d.media.clip.url.includes('?st='), 'clip not signed');
    return `overlays=${d.overlays?.length ?? 0} calls=${d.escalation?.calls?.length ?? 0} circles_ready=${d.media.clip_circles.ready}`;
  });
  await check('signed media: range 206, HEAD, unsigned 401', async () => {
    const d = await api.event(eventId);
    const url = api.url(d.media.clip_circles.ready ? d.media.clip_circles.url : d.media.clip.url);
    const r = await fetch(url, { headers: { Range: 'bytes=0-1023' } });
    assert(r.status === 206, `range gave ${r.status}`);
    const buf = new Uint8Array(await r.arrayBuffer());
    assert(buf.length === 1024, `got ${buf.length} bytes`);
    const h = await fetch(url, { method: 'HEAD' });
    assert(h.ok, `HEAD ${h.status}`);
    const unsigned = await fetch(url.split('?')[0]);
    assert(unsigned.status === 401, `unsigned gave ${unsigned.status}`);
    const th = await fetch(api.url(d.media.thumb.url));
    assert(th.ok && th.headers.get('content-type')?.startsWith('image/'), `thumb ${th.status}`);
    return `len=${h.headers.get('content-length')} thumb ok`;
  });
  await check('ack / dismiss / reopen', async () => {
    const before = (await api.event(eventId)).status;
    assert((await api.ack(eventId)).status === 'acknowledged', 'ack');
    assert((await api.dismiss(eventId)).status === 'dismissed', 'dismiss');
    await api.setStatus(eventId, before);
    return `restored to ${before}`;
  });
}

await check('WebSocket via one-time ticket', async () => {
  const t = await api.wsTicket();
  const url = api.url(t.url).replace(/^http/, 'ws');
  const hello = await new Promise<string>((resolve, reject) => {
    const ws = new WebSocket(url);
    const to = setTimeout(() => reject(new Error('no hello in 5 s')), 5000);
    ws.onmessage = (ev) => {
      clearTimeout(to);
      ws.close();
      resolve(String(ev.data));
    };
    ws.onerror = () => reject(new Error('ws error'));
  });
  assert(JSON.parse(hello).type === 'hello', 'first message is not hello');
  // The backend closes before accept, so the handshake is refused (HTTP 403 -> client sees 1006, or
  // 4401 on servers that pass the close code through). Either way no message may arrive.
  const reuse = await new Promise<{ code: number; got: boolean }>((resolve) => {
    let got = false;
    const ws = new WebSocket(url);
    ws.onmessage = () => {
      got = true;
      ws.close();
    };
    ws.onclose = (ev) => resolve({ code: ev.code, got });
  });
  assert(!reuse.got && reuse.code !== 1000, `reused ticket was accepted (code ${reuse.code})`);
  return `hello received; reused ticket refused (close ${reuse.code})`;
});

await check('POST /api/devices rejects a non-Expo token', async () => {
  try {
    await api.registerDevice('not-a-real-token-123');
    throw new Error('accepted');
  } catch (e) {
    assert(e instanceof ApiError && e.status === 422, String(e));
  }
});

await check('logout revokes the session', async () => {
  const rt = store.value;
  await api.logout();
  const r = await fetch(`${BASE}/api/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: rt }),
  });
  assert(r.status === 401, `refresh after logout gave ${r.status}`);
});

console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
