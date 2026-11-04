// Session context: owns the ApiClient, the signed-in user and the server URL.
import Constants from 'expo-constants';
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { ApiClient, ApiError } from './api';
import { prefs, secureTokenStore } from './storage';
import type { SessionUser } from './types';
import { normalizeServerUrl } from './url';

type Phase = 'loading' | 'signedOut' | 'signedIn' | 'offline';

interface SessionValue {
  phase: Phase;
  api: ApiClient | null;
  user: SessionUser | null;
  server: string;
  defaultServer: string;
  expiredReason: string | null;
  signIn(server: string, username: string, password: string): Promise<void>;
  signOut(): Promise<void>;
  retry(): Promise<void>;
}

const Ctx = createContext<SessionValue | null>(null);

export const DEFAULT_SERVER: string = (Constants.expoConfig?.extra?.defaultServerUrl as string | undefined) ?? '';

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [phase, setPhase] = useState<Phase>('loading');
  const [api, setApi] = useState<ApiClient | null>(null);
  const [user, setUser] = useState<SessionUser | null>(null);
  const [server, setServer] = useState('');
  const [expiredReason, setExpiredReason] = useState<string | null>(null);
  const apiRef = useRef<ApiClient | null>(null);

  const makeClient = useCallback((base: string) => {
    const client = new ApiClient({
      baseUrl: base,
      store: secureTokenStore,
      onUser: (u) => {
        if (apiRef.current === client) setUser(u);
      },
      onSessionExpired: (reason) => {
        if (apiRef.current !== client) return;
        setExpiredReason(reason);
        setPhase('signedOut');
      },
    });
    apiRef.current = client;
    setApi(client);
    return client;
  }, []);

  const boot = useCallback(async () => {
    setPhase('loading');
    const saved = (await prefs.getServer()) || '';
    setServer(saved || DEFAULT_SERVER);
    if (!saved) {
      setPhase('signedOut');
      return;
    }
    const client = makeClient(saved);
    try {
      const u = await client.restore();
      setUser(u);
      setPhase(u ? 'signedIn' : 'signedOut');
    } catch (e) {
      setPhase(e instanceof ApiError && e.isNetwork ? 'offline' : 'signedOut');
    }
  }, [makeClient]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- async fetch on mount; state updates happen after awaits
    void boot();
  }, [boot]);

  const signIn = useCallback(
    async (serverInput: string, username: string, password: string) => {
      const base = normalizeServerUrl(serverInput);
      const client = makeClient(base);
      await client.health(); // friendly error when the address is wrong or the PC is off
      const u = await client.login(username.trim(), password);
      await prefs.setServer(base);
      await prefs.setUsername(username.trim());
      setServer(base);
      setUser(u);
      setExpiredReason(null);
      setPhase('signedIn');
    },
    [makeClient]
  );

  const signOut = useCallback(async () => {
    const client = apiRef.current;
    setPhase('signedOut');
    setUser(null);
    if (client) await client.logout();
  }, []);

  const value = useMemo<SessionValue>(
    () => ({
      phase,
      api,
      user,
      server,
      defaultServer: DEFAULT_SERVER,
      expiredReason,
      signIn,
      signOut,
      retry: boot,
    }),
    [phase, api, user, server, expiredReason, signIn, signOut, boot]
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSession(): SessionValue {
  const v = useContext(Ctx);
  if (!v) throw new Error('useSession outside SessionProvider');
  return v;
}

/** The ApiClient for screens that are only reachable while signed in. */
export function useApi(): ApiClient {
  const { api } = useSession();
  if (!api) throw new Error('not signed in');
  return api;
}
