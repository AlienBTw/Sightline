// Alerts on the phone.
// 1) Remote push (Expo push service -> FCM): works only in a build that has Firebase credentials
//    (google-services.json) and an EAS projectId. Then the backend's push reaches the phone even when
//    the app is closed.
// 2) Without FCM, the app still alerts from the WebSocket: an in-app banner while open, and a local
//    notification while it is in the background (as long as Android keeps the process alive).
// Tapping either kind opens the event (payload carries only event_id, factor and risk).
import Constants from 'expo-constants';
import * as Device from 'expo-device';
import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import type { ApiClient } from './api';
import { factorInfo, headlineFactor, isExperimental } from './factors';
import { prefs } from './storage';
import type { EventSummary } from './types';

export const CHANNEL_ID = 'alerts';

let configured = false;
let remotePushActive = false;

export function configureNotifications() {
  if (configured) return;
  configured = true;
  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldShowBanner: true,
      shouldShowList: true,
      shouldPlaySound: true,
      shouldSetBadge: false,
    }),
  });
  if (Platform.OS === 'android') {
    void Notifications.setNotificationChannelAsync(CHANNEL_ID, {
      name: 'Hazard alerts',
      description: 'A camera flagged a possible hazard',
      importance: Notifications.AndroidImportance.HIGH,
      vibrationPattern: [0, 250, 150, 250],
      lightColor: '#c45c26',
      lockscreenVisibility: Notifications.AndroidNotificationVisibility.PUBLIC,
      showBadge: true,
    }).catch(() => {});
  }
}

export async function ensurePermission(ask = true): Promise<boolean> {
  const cur = await Notifications.getPermissionsAsync();
  if (cur.granted) return true;
  if (!ask || !cur.canAskAgain) return false;
  const r = await Notifications.requestPermissionsAsync();
  return r.granted;
}

export type PushStatus =
  | { state: 'registered'; token: string }
  | { state: 'local_only'; reason: string }
  | { state: 'denied'; reason: string };

function projectId(): string | undefined {
  const extra = Constants.expoConfig?.extra as { eas?: { projectId?: string } } | undefined;
  return extra?.eas?.projectId ?? Constants.easConfig?.projectId ?? undefined;
}

/** Try to register for remote push; falls back to local-only alerts with a reason. */
export async function registerForPush(api: ApiClient): Promise<PushStatus> {
  remotePushActive = false;
  if (!(await ensurePermission(true))) {
    return { state: 'denied', reason: 'Notifications are turned off for Sightline in Android settings.' };
  }
  if (!Device.isDevice) return { state: 'local_only', reason: 'Remote push needs a real phone (emulator).' };
  const pid = projectId();
  if (!pid) {
    return {
      state: 'local_only',
      reason: 'This build has no Expo project / Firebase setup, so alerts come over the live connection.',
    };
  }
  try {
    const { data: token } = await Notifications.getExpoPushTokenAsync({ projectId: pid });
    await api.registerDevice(token, Platform.OS);
    await prefs.setPushToken(token);
    remotePushActive = true;
    return { state: 'registered', token };
  } catch (e) {
    return {
      state: 'local_only',
      reason: `Remote push unavailable (${e instanceof Error ? e.message.slice(0, 120) : 'unknown error'}).`,
    };
  }
}

export async function unregisterPush(api: ApiClient) {
  const token = await prefs.getPushToken();
  remotePushActive = false;
  if (token) {
    await api.unregisterDevice(token).catch(() => {});
    await prefs.setPushToken(null);
  }
}

// ------------------------------------------------------------------ in-app banner bus
type BannerListener = (ev: EventSummary) => void;
const bannerListeners = new Set<BannerListener>();

export function onInAppAlert(l: BannerListener) {
  bannerListeners.add(l);
  return () => {
    bannerListeners.delete(l);
  };
}

export function alertText(ev: Pick<EventSummary, 'top_factor' | 'factors' | 'risk_max'>) {
  const f = headlineFactor(ev.top_factor, ev.factors);
  const name = f ? factorInfo(f).name : 'Hazard';
  const risk = Math.round((ev.risk_max ?? 0) * 100);
  const exp = isExperimental(f) ? ' (experimental)' : '';
  return { title: `Hazard: ${name}${exp}`, body: `Risk ${risk}%. Tap to review the flagged clip.` };
}

/** Called for every new alerting event from the WebSocket. */
export async function notifyNewEvent(ev: EventSummary, foreground: boolean) {
  if (foreground) {
    bannerListeners.forEach((l) => l(ev));
    return;
  }
  if (remotePushActive) return; // the backend's push already covers the background case
  try {
    if (!(await ensurePermission(false))) return;
    const { title, body } = alertText(ev);
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: { event_id: ev.event_id, factor: ev.top_factor ?? '', risk: ev.risk_max ?? 0 },
        sound: 'default',
      },
      trigger: Platform.OS === 'android' ? { channelId: CHANNEL_ID } : null,
    });
  } catch {
    // notifications are best effort
  }
}

export async function sendLocalTest() {
  if (!(await ensurePermission(true))) return false;
  await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Sightline test alert',
      body: 'Local notifications work on this phone.',
      data: {},
    },
    trigger: Platform.OS === 'android' ? { channelId: CHANNEL_ID } : null,
  });
  return true;
}

/** event_id from a tapped notification (remote push or local). */
export function eventIdFromResponse(r: Notifications.NotificationResponse | null | undefined): string | null {
  const id = (r?.notification.request.content.data as { event_id?: unknown } | undefined)?.event_id;
  return typeof id === 'string' && /^evt_[A-Za-z0-9_-]{1,120}$/.test(id) && id !== 'evt_test' ? id : null;
}

// ------------------------------------------------------------------ push status store (Settings)
let currentStatus: PushStatus | null = null;
const statusListeners = new Set<() => void>();

export function setPushStatus(s: PushStatus | null) {
  currentStatus = s;
  statusListeners.forEach((l) => l());
}

export function getPushStatus() {
  return currentStatus;
}

export function subscribePushStatus(l: () => void) {
  statusListeners.add(l);
  return () => {
    statusListeners.delete(l);
  };
}
