import * as Notifications from 'expo-notifications';
import { router, Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import {
  IBMPlexMono_500Medium,
} from '@expo-google-fonts/ibm-plex-mono';
import {
  IBMPlexSans_400Regular,
  IBMPlexSans_500Medium,
  IBMPlexSans_600SemiBold,
} from '@expo-google-fonts/ibm-plex-sans';
import { useFonts } from 'expo-font';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useLayoutEffect, useRef } from 'react';
import { View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { InAppAlert } from '../components/InAppAlert';
import {
  configureNotifications,
  eventIdFromResponse,
  registerForPush,
  setPushStatus,
} from '../lib/notifications';
import { RealtimeProvider } from '../lib/realtime';
import { SessionProvider, useSession } from '../lib/session';
import { C, FONT } from '../lib/theme';

void SplashScreen.preventAutoHideAsync().catch(() => {});
configureNotifications();

function openEvent(id: string) {
  router.push({ pathname: '/event/[id]', params: { id } });
}

/** Push/local notification taps -> event detail (after sign-in if needed). */
function NotificationRouter() {
  const { phase, api } = useSession();
  const pending = useRef<string | null>(null);
  const signedIn = phase === 'signedIn';
  const signedRef = useRef(signedIn);
  useLayoutEffect(() => {
    signedRef.current = signedIn;
  }, [signedIn]);

  useEffect(() => {
    const first = eventIdFromResponse(Notifications.getLastNotificationResponse());
    if (first) {
      pending.current = first;
      Notifications.clearLastNotificationResponse();
    }
    const sub = Notifications.addNotificationResponseReceivedListener((r) => {
      const id = eventIdFromResponse(r);
      if (!id) return;
      pending.current = id;
      if (signedRef.current) {
        pending.current = null;
        openEvent(id);
      }
    });
    return () => sub.remove();
  }, []);

  useEffect(() => {
    if (!signedIn) return;
    if (pending.current) {
      const id = pending.current;
      pending.current = null;
      setTimeout(() => openEvent(id), 50);
    }
  }, [signedIn]);

  useEffect(() => {
    if (!signedIn || !api) return;
    let cancelled = false;
    registerForPush(api)
      .then((s) => !cancelled && setPushStatus(s))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [signedIn, api]);

  return null;
}

function RootStack() {
  const { phase } = useSession();
  const signedIn = phase === 'signedIn';

  useEffect(() => {
    if (phase !== 'loading') void SplashScreen.hideAsync().catch(() => {});
  }, [phase]);

  if (phase === 'loading') return <View style={{ flex: 1, backgroundColor: C.bg }} />;

  return (
    <>
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: C.bg },
          headerTintColor: C.text,
          headerTitleStyle: { fontFamily: FONT.sansSem, fontWeight: '600' },
          headerShadowVisible: false,
          contentStyle: { backgroundColor: C.bg },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Protected guard={signedIn}>
          <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
          <Stack.Screen name="camera/[id]" options={{ title: 'Live view' }} />
          <Stack.Screen name="event/[id]" options={{ title: 'Alert' }} />
        </Stack.Protected>
        <Stack.Protected guard={!signedIn}>
          <Stack.Screen name="login" options={{ headerShown: false, animation: 'fade' }} />
        </Stack.Protected>
      </Stack>
      {signedIn ? <InAppAlert /> : null}
    </>
  );
}

export default function RootLayout() {
  const [fontsReady] = useFonts({
    IBMPlexSans_400Regular,
    IBMPlexSans_500Medium,
    IBMPlexSans_600SemiBold,
    IBMPlexMono_500Medium,
  });
  if (!fontsReady) return <View style={{ flex: 1, backgroundColor: C.bg }} />;
  return (
    <SafeAreaProvider style={{ backgroundColor: C.bg }}>
      <StatusBar style="dark" />
      <SessionProvider>
        <RealtimeProvider>
          <NotificationRouter />
          <RootStack />
        </RealtimeProvider>
      </SessionProvider>
    </SafeAreaProvider>
  );
}
