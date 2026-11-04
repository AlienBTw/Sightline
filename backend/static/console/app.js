"use strict";
// Sightline dev console. Same login and API as the mobile app. No inline scripts (strict CSP).
// Access token in memory only; refresh token in sessionStorage (cleared when the tab closes).
const $ = (id) => document.getElementById(id);
const S = { access: null, user: null, offset: 0, selected: null, ws: null, camTimer: null };

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}

async function refresh() {
  const rt = sessionStorage.getItem("sl_refresh");
  if (!rt) return false;
  const r = await fetch("/api/auth/refresh", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ refresh_token: rt }) });
  if (!r.ok) { sessionStorage.removeItem("sl_refresh"); return false; }
  setTokens(await r.json());
  return true;
}

function setTokens(t) {
  S.access = t.access_token; S.user = t.user;
  sessionStorage.setItem("sl_refresh", t.refresh_token);
}

async function api(path, opts = {}, retry = true) {
  const headers = Object.assign({}, opts.headers || {}, S.access ? { Authorization: "Bearer " + S.access } : {});
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (r.status === 401 && retry && await refresh()) return api(path, opts, false);
  if (r.status === 401) { showLogin(); throw new Error("signed out"); }
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : null;
  if (!r.ok) throw new Error((body && body.detail) ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : r.statusText);
  return body;
}

function showLogin() {
  S.access = null; $("login-view").classList.remove("hidden"); $("app-view").classList.add("hidden");
  $("logout").classList.add("hidden"); $("who").textContent = "";
  if (S.ws) S.ws.close(); clearInterval(S.camTimer);
}

async function showApp() {
  const me = await api("/api/auth/me");
  S.user = { username: me.username, role: me.role };
  $("who").textContent = `${me.username} (${me.role})`;
  $("login-view").classList.add("hidden"); $("app-view").classList.remove("hidden"); $("logout").classList.remove("hidden");
  $("rtsp-card").classList.toggle("hidden", me.role !== "admin");
  await loadCameras(); await loadEvents(true); connectWs();
  clearInterval(S.camTimer); S.camTimer = setInterval(loadCameras, 60000);   // WebSocket pushes changes; this is a slow safety net
}

// ---------------------------------------------------------------- cameras + live preview
const streams = {};   // camera id -> <img> showing its MJPEG stream (kept across list refreshes)
async function loadCameras() {
  let d; try { d = await api("/api/cameras"); } catch (e) { return; }
  $("capacity").textContent = `${d.capacity.active}/${d.capacity.max} running on ${d.capacity.gpus.join(", ")}`;
  const box = $("cameras"); box.replaceChildren();
  const sel = $("f-camera"); const cur = sel.value; sel.replaceChildren(el("option", { value: "" }, "All cameras"));
  const seen = new Set();
  for (const c of d.items) {
    sel.append(el("option", { value: c.id }, c.name));
    const st = c.status.state; const running = st === "running";
    let img = streams[c.id];
    if (running) {
      seen.add(c.id);
      if (!img) {   // one long MJPEG request per camera (1 fps, circles drawn server-side), not polling
        img = el("img", { alt: c.name, src: c.preview.mjpeg_url + "&fps=1&w=480" });
        img.addEventListener("error", () => { delete streams[c.id]; });
        streams[c.id] = img;
      }
    } else img = el("img", { alt: "" });
    const btn = running || st === "loading_model"
      ? el("button", { class: "ghost", onclick: () => camAction(c.id, "stop") }, "Stop")
      : el("button", { class: "teal", onclick: () => camAction(c.id, "start") }, "Start");
    const meta = el("div", { class: "meta" }, el("span", { class: "name" }, c.name),
      el("span", { class: "badge " + st }, st), el("span", { class: "badge" }, c.kind),
      ...(c.status.active_factors || []).map((f) => el("span", { class: "badge alert" }, hazardName(f))),
      el("span", { class: "muted" }, c.status.device ? `${c.status.device} ${c.status.fps || 0} fps` : ""), btn);
    if (S.user.role === "admin") meta.append(el("button", { class: "ghost", onclick: () => delCam(c.id) }, "Delete"));
    if (c.status.last_error) meta.append(el("span", { class: "error" }, c.status.last_error));
    box.append(el("div", { class: "cam" }, img, meta));
  }
  for (const id of Object.keys(streams)) if (!seen.has(id)) { streams[id].src = ""; delete streams[id]; }
  sel.value = cur;
  if (!d.items.length) box.append(el("p", { class: "muted" }, "Nothing is connected yet. An admin adds an RTSP camera."));
}

const HAZARD_NAMES = {
  F1: "Hand in danger zone", F2: "No head protection", F3: "No eye/face protection",
  F4: "Wrong glove state", F5: "No hi-vis vest", F6: "Fall or slip",
  F7: "Unstable load", F8: "Person in vehicle path", F9: "Open guard or panel",
  F10: "Fire, smoke or sparks", F11: "Abnormal machine behaviour",
};
function hazardName(id) { return HAZARD_NAMES[id] || id || "Hazard"; }
async function camAction(id, action) { try { await api(`/api/cameras/${id}/${action}`, { method: "POST" }); } catch (e) { alert(e.message); } setTimeout(loadCameras, 600); }
async function delCam(id) { if (!confirm("Delete this camera? Its events are kept.")) return; try { await api(`/api/cameras/${id}`, { method: "DELETE" }); } catch (e) { alert(e.message); } loadCameras(); }

async function addRtsp(ev) {
  ev.preventDefault();
  try {
    const c = await api("/api/cameras", { method: "POST", json: { name: $("rtsp-name").value, uri: $("rtsp-uri").value } });
    $("rtsp-msg").textContent = `Added ${c.name}` + (c.started ? " and started." : `; not started: ${c.start_error}`);
    $("rtsp-form").reset(); loadCameras();
  } catch (e) { $("rtsp-msg").textContent = e.message; }
}

// ---------------------------------------------------------------- events + player
async function loadEvents(reset) {
  if (reset) { S.offset = 0; $("events").replaceChildren(); }
  const q = new URLSearchParams({ limit: 20, offset: S.offset });
  if ($("f-status").value) q.set("status", $("f-status").value);
  if ($("f-camera").value) q.set("camera_id", $("f-camera").value);
  let d; try { d = await api("/api/events?" + q); } catch (e) { return; }
  for (const e of d.items) $("events").append(evRow(e));
  S.offset = d.next_offset || S.offset + d.items.length;
  $("more").classList.toggle("hidden", d.next_offset == null);
  if (!d.total) $("events").append(el("p", { class: "muted" }, "No events yet."));
}

function evRow(e, prepend) {
  const row = el("div", { class: "ev", "data-id": e.event_id, onclick: () => openEvent(e.event_id) },
    el("img", { src: e.thumb_url, alt: "", loading: "lazy" }),
    el("div", {}, el("div", { class: "t" }, e.top_factor_label || hazardName(e.top_factor)),
      el("div", {}, el("span", { class: "risk" }, `risk ${Math.round((e.risk_max || 0) * 100)}%`), "  ", el("span", { class: "badge" }, e.status)),
      el("div", { class: "muted" }, `${e.camera_id} · ${new Date(e.started_at).toLocaleString()}`)));
  return row;
}

async function openEvent(id) {
  S.selected = id;
  document.querySelectorAll(".ev").forEach((n) => n.classList.toggle("sel", n.dataset.id === id));
  let d; try { d = await api(`/api/events/${id}`); } catch (e) { $("detail").textContent = e.message; return; }
  const video = el("video", { controls: "", playsinline: "", preload: "metadata" });
  const pick = (k) => { video.src = d.media[k].url; video.play().catch(() => {}); };
  pick(d.media.clip_circles.ready ? "clip_circles" : "clip");
  const esc = (d.escalation && d.escalation.calls) || [];
  $("detail").replaceChildren(
    el("h3", {}, d.top_factor_label || hazardName(d.top_factor)),
    el("div", { class: "muted" }, `${d.camera_id} · ${d.started_at} → ${d.ended_at || ""} · risk ${Math.round(d.risk_max * 100)}% · ${(d.factors || []).map(hazardName).join(", ")}`),
    video,
    el("div", { class: "row" },
      el("button", { class: "ghost", onclick: () => pick("clip_circles") }, "With circles"),
      el("button", { class: "ghost", onclick: () => pick("clip") }, "Raw clip"),
      el("button", { class: "teal", onclick: () => setStatus(id, "ack") }, "Acknowledge"),
      el("button", { class: "ghost", onclick: () => setStatus(id, "dismiss") }, "Dismiss"),
      el("span", { class: "badge" }, d.status)),
    el("div", { class: "muted" }, `Escalations: ${esc.length ? esc.map((c) => `${hazardName(c.factor)} (${c.reason})`).join(", ") : "none"}`),
    el("pre", {}, JSON.stringify({ segments: d.segments, escalation: d.escalation && d.escalation.calls, localization: d.localization }, null, 1)));
}

async function setStatus(id, action) {
  try { await api(`/api/events/${id}/${action}`, { method: "POST" }); } catch (e) { alert(e.message); }
  openEvent(id); loadEvents(true);
}

// ---------------------------------------------------------------- real-time
async function connectWs() {
  if (S.ws && S.ws.readyState <= 1) return;
  let t; try { t = await api("/api/ws/ticket", { method: "POST" }); } catch (e) { return; }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}${t.url}`);
  S.ws = ws;
  ws.onopen = () => { $("ws-state").textContent = "live: connected"; };
  ws.onclose = () => { $("ws-state").textContent = "live: reconnecting"; if (S.access) setTimeout(connectWs, 3000); };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "event") {
      loadEvents(true); loadCameras();
    } else if (msg.type === "camera" || msg.type === "alert_state" || msg.type.startsWith("camera_")) {
      clearTimeout(S.camDebounce); S.camDebounce = setTimeout(loadCameras, 500);
    }
  };
}

document.addEventListener("DOMContentLoaded", async () => {
  $("login-form").addEventListener("submit", async (ev) => {
    ev.preventDefault(); $("login-error").textContent = "";
    const r = await fetch("/api/auth/login", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: $("username").value, password: $("password").value }) });
    const b = await r.json().catch(() => ({}));
    if (!r.ok) { $("login-error").textContent = b.detail || "Sign-in failed"; return; }
    $("password").value = ""; setTokens(b); showApp();
  });
  $("logout").addEventListener("click", async () => {
    const rt = sessionStorage.getItem("sl_refresh");
    if (rt) await fetch("/api/auth/logout", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ refresh_token: rt }) });
    sessionStorage.removeItem("sl_refresh"); showLogin();
  });
  $("rtsp-form").addEventListener("submit", addRtsp);
  $("refresh-events").addEventListener("click", () => loadEvents(true));
  $("f-status").addEventListener("change", () => loadEvents(true));
  $("f-camera").addEventListener("change", () => loadEvents(true));
  $("more").addEventListener("click", () => loadEvents(false));
  if (await refresh()) showApp(); else showLogin();
});
