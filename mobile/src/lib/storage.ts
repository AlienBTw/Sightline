// Persistent settings. The refresh token goes to expo-secure-store (Android Keystore-backed);
// nothing sensitive is kept in plain storage.
import * as SecureStore from 'expo-secure-store';
import type { TokenStore } from './api';

const K = { server: 'sl_server', refresh: 'sl_refresh', username: 'sl_username', pushToken: 'sl_push_token' };
const OPTS: SecureStore.SecureStoreOptions = { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY };

async function get(key: string) {
  try {
    return await SecureStore.getItemAsync(key, OPTS);
  } catch {
    return null; // e.g. keystore reset after a backup restore: treat as signed out
  }
}

async function set(key: string, value: string | null) {
  if (value == null) await SecureStore.deleteItemAsync(key, OPTS).catch(() => {});
  else await SecureStore.setItemAsync(key, value, OPTS);
}

export const secureTokenStore: TokenStore = {
  getRefreshToken: () => get(K.refresh),
  setRefreshToken: (t) => set(K.refresh, t),
};

export const prefs = {
  getServer: () => get(K.server),
  setServer: (v: string | null) => set(K.server, v),
  getUsername: () => get(K.username),
  setUsername: (v: string | null) => set(K.username, v),
  getPushToken: () => get(K.pushToken),
  setPushToken: (v: string | null) => set(K.pushToken, v),
};
