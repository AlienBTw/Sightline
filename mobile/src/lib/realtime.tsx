// One WebSocket per signed-in app session: POST /api/ws/ticket (one-time, 60 s) then WS /api/ws?ticket=.
// The access token never appears in a URL. Reconnects with backoff, and when the app comes back to
// the foreground. Screens subscribe with useRealtime(listener).
import React, { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { AppState } from 'react-native';
import { ApiError } from './api';
import { isAlerting } from './factors';
import { notifyNewEvent } from './notifications';
import { useSession } from './session';
import type { WsMessage } from './types';
import { wsUrl } from './url';

type Listener = (m: WsMessage) => void;

interface RealtimeValue {
  connected: boolean;
  cameraFactors: Record<string, string[]>;
  subscribe(l: Listener): () => void;
}

const Ctx = createContext<RealtimeValue | null>(null);

export function RealtimeProvider({ children }: { children: React.ReactNode }) {
  const { api, phase } = useSession();
  const [connected, setConnected] = useState(false);
  const [cameraFactors, setCameraFactors] = useState<Record<string, string[]>>({});
  const listeners = useRef(new Set<Listener>());
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attempt = useRef(0);
  const active = phase === 'signedIn' && !!api;

  const emit = useCallback((m: WsMessage) => {
    if (m.type === 'hello') {
      setCameraFactors(Object.fromEntries(m.cameras.map((c) => [c.camera_id, c.active_factors ?? []])));
    } else if (m.type === 'alert_state') {
      setCameraFactors((prev) => ({ ...prev, [m.camera_id]: m.factors ?? [] }));
    } else if (m.type === 'event' && isAlerting(m.event?.factors, m.event?.verified_by)) {
      void notifyNewEvent(m.event, AppState.currentState === 'active');
    }
    listeners.current.forEach((l) => {
      try {
        l(m);
      } catch {
        // a broken screen listener must not kill the socket
      }
    });
  }, []);

  useEffect(() => {
    if (!active || !api) return;
    let stopped = false;

    const clearTimers = () => {
      if (retryRef.current) clearTimeout(retryRef.current);
      if (idleRef.current) clearTimeout(idleRef.current);
      retryRef.current = idleRef.current = null;
    };

    const schedule = () => {
      if (stopped) return;
      const delay = Math.min(30000, 1000 * 2 ** Math.min(attempt.current, 5)) + Math.random() * 500;
      attempt.current += 1;
      retryRef.current = setTimeout(connect, delay);
    };

    const kick = () => {
      if (idleRef.current) clearTimeout(idleRef.current);
      // the server pings every 20 s; silence for 50 s means a dead link (NAT / tunnel drop)
      idleRef.current = setTimeout(() => wsRef.current?.close(), 50000);
    };

    async function connect() {
      if (stopped || !api) return;
      if (AppState.currentState !== 'active' && attempt.current > 3) return; // resume on foreground
      try {
        const t = await api.wsTicket();
        if (stopped) return;
        const ws = new WebSocket(wsUrl(api.baseUrl, t.url));
        wsRef.current = ws;
        ws.onopen = () => {
          attempt.current = 0;
          setConnected(true);
          kick();
        };
        ws.onmessage = (ev) => {
          kick();
          try {
            const m = JSON.parse(String(ev.data)) as WsMessage;
            if (m.type !== 'ping' && m.type !== 'pong') emit(m);
          } catch {
            // ignore malformed frames
          }
        };
        ws.onerror = () => {};
        ws.onclose = () => {
          if (wsRef.current === ws) wsRef.current = null;
          setConnected(false);
          schedule();
        };
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) return; // session ended -> login screen
        setConnected(false);
        schedule();
      }
    }

    attempt.current = 0;
    void connect();
    const sub = AppState.addEventListener('change', (s) => {
      if (s === 'active' && !wsRef.current) {
        clearTimers();
        attempt.current = 0;
        void connect();
      }
    });
    return () => {
      stopped = true;
      sub.remove();
      clearTimers();
      const ws = wsRef.current;
      wsRef.current = null;
      ws?.close();
      setConnected(false);
    };
  }, [active, api, emit]);

  const subscribe = useCallback((l: Listener) => {
    listeners.current.add(l);
    return () => {
      listeners.current.delete(l);
    };
  }, []);

  return <Ctx.Provider value={{ connected, cameraFactors, subscribe }}>{children}</Ctx.Provider>;
}

export function useRealtimeState() {
  const v = useContext(Ctx);
  if (!v) throw new Error('useRealtimeState outside RealtimeProvider');
  return v;
}

/** Subscribe to WebSocket messages for the lifetime of the component. */
export function useRealtime(listener: Listener) {
  const { subscribe } = useRealtimeState();
  const ref = useRef(listener);
  useLayoutEffect(() => {
    ref.current = listener;
  }, [listener]);
  useEffect(() => subscribe((m) => ref.current(m)), [subscribe]);
}
