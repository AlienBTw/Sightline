import { Ionicons } from '@expo/vector-icons';
import { router, Stack, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Alert, AppState, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { CameraOptionsSheet } from '../../components/CameraOptionsSheet';
import { EventRow } from '../../components/EventRow';
import { MjpegView } from '../../components/MjpegView';
import { Button, EmptyState, ErrorState, FactorChip, Loading, StatusDot, Txt } from '../../components/ui';
import { visibleFactors } from '../../lib/factors';
import { useRealtime, useRealtimeState } from '../../lib/realtime';
import { useApi, useSession } from '../../lib/session';
import { C, CAMERA_STATE, R, S } from '../../lib/theme';
import type { Camera, EventSummary } from '../../lib/types';
import { absUrl } from '../../lib/url';
import { errMsg, useAsync } from '../../lib/useAsync';

const SESSION_S = 290; // the backend ends one MJPEG response after 300 s (ngrok data budget)

export default function LiveView() {
  const { id, name } = useLocalSearchParams<{ id: string; name?: string }>();
  const api = useApi();
  const { user } = useSession();
  const { cameraFactors } = useRealtimeState();
  const cam = useAsync<Camera>(() => api.camera(id), [api, id]);
  const events = useAsync(() => api.events({ camera_id: id, limit: 10 }), [api, id]);
  const [focused, setFocused] = useState(true);
  const [appActive, setAppActive] = useState(AppState.currentState === 'active');
  const [session, setSession] = useState(0);
  const [expired, setExpired] = useState(false);
  const [gotFrame, setGotFrame] = useState(false);
  const [streamErr, setStreamErr] = useState(false);
  const [busy, setBusy] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const isAdmin = user?.role === 'admin';

  useFocusEffect(
    useCallback(() => {
      setFocused(true);
      return () => setFocused(false);
    }, [])
  );
  useEffect(() => {
    const sub = AppState.addEventListener('change', (s) => setAppActive(s === 'active'));
    return () => sub.remove();
  }, []);

  const c = cam.data;
  const running = c?.status.state === 'running';
  const watching = running && focused && appActive && !expired;

  // One viewing session = one long MJPEG response; offer "keep watching" when it ends.
  useEffect(() => {
    if (!watching) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset per viewing session
    setGotFrame(false);
    setStreamErr(false);
    const t = setTimeout(() => setExpired(true), SESSION_S * 1000);
    return () => clearTimeout(t);
  }, [watching, session]);

  const keepWatching = useCallback(async () => {
    await cam.reload('silent'); // fresh signed link
    setExpired(false);
    setSession((s) => s + 1);
  }, [cam]);

  useRealtime((m) => {
    if (m.type === 'camera' && m.camera_id === id) void cam.reload('silent');
    if (m.type === 'event' && m.event.camera_id === id) void events.reload('silent');
    if (m.type === 'event_status')
      events.setData((d) =>
        d ? { ...d, items: d.items.map((e) => (e.event_id === m.event_id ? { ...e, status: m.status } : e)) } : d
      );
  });

  const canControl = isAdmin || c?.kind === 'virtual';
  async function toggle() {
    if (!c) return;
    setBusy(true);
    try {
      if (running || c.status.state === 'starting' || c.status.state === 'loading_model') await api.stopCamera(c.id);
      else await api.startCamera(c.id);
      await cam.reload('silent');
      setTimeout(() => void cam.reload('silent'), 9000);
    } catch (e) {
      Alert.alert('Camera', errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove(target: Camera) {
    try {
      await api.deleteCamera(target.id);
      setMenuOpen(false);
      router.replace('/');
    } catch (e) {
      Alert.alert('Could not delete', errMsg(e));
    }
  }

  if (cam.loading && !c) return <Loading label="Connecting to camera…" />;
  if (cam.error && !c) return <ErrorState message={cam.error} onRetry={() => void cam.reload()} />;
  if (!c) return null;

  const st = CAMERA_STATE[c.status.state] ?? { label: c.status.state, color: C.muted };
  const factors = visibleFactors(cameraFactors[c.id] ?? c.status.active_factors);
  const starting = c.status.state === 'starting' || c.status.state === 'loading_model';

  const header = (
    <View style={{ gap: S.md }}>
      <Pressable
        onLongPress={() => setMenuOpen(true)}
        delayLongPress={380}
        accessibilityRole="image"
        accessibilityLabel={`${c.name} live picture`}
        accessibilityHint="Hold for camera options"
        style={styles.video}
      >
        {watching ? (
          <>
            <MjpegView
              key={session}
              url={absUrl(api.baseUrl, c.preview.mjpeg_url)}
              onFrame={() => setGotFrame(true)}
              onError={() => setStreamErr(true)}
            />
            {!gotFrame && !streamErr ? (
              <View style={styles.center} pointerEvents="none">
                <ActivityIndicator color={C.accent} />
              </View>
            ) : null}
            {streamErr ? (
              <View style={styles.center}>
                <Txt variant="muted">The live stream was interrupted.</Txt>
                <Button compact kind="secondary" title="Reconnect" icon="refresh" onPress={keepWatching} style={{ marginTop: S.sm }} />
              </View>
            ) : null}
            <View style={styles.liveTag} pointerEvents="none">
              <View style={styles.liveDot} />
              <Text style={styles.liveText}>LIVE · 1 fps</Text>
            </View>
          </>
        ) : expired && running ? (
          <View style={styles.center}>
            <Ionicons name="pause-circle-outline" size={40} color={C.muted} />
            <Txt variant="muted" style={{ marginVertical: S.sm }}>
              Live view paused to save data.
            </Txt>
            <Button title="Keep watching" icon="play" onPress={keepWatching} />
          </View>
        ) : (
          <View style={styles.center}>
            {starting ? <ActivityIndicator color={C.accent} /> : <Ionicons name="videocam-off-outline" size={40} color={C.faint} />}
            <Txt variant="muted" style={{ marginTop: S.sm }}>
              {starting ? 'Loading the model (about 10 s)…' : 'This camera is not running.'}
            </Txt>
          </View>
        )}
      </Pressable>
      <Txt variant="small">Hold the picture for options.</Txt>

      <View style={styles.statusRow}>
        <View style={styles.state}>
          <StatusDot color={st.color} />
          <Txt style={{ fontWeight: '700' }}>{st.label}</Txt>
          {running && c.status.fps ? <Txt variant="small">· {c.status.fps.toFixed(1)} fps</Txt> : null}
        </View>
        {canControl ? (
          <Button
            compact
            kind={running || starting ? 'secondary' : 'primary'}
            icon={running || starting ? 'stop' : 'play'}
            title={running || starting ? 'Stop' : 'Start'}
            loading={busy}
            onPress={toggle}
          />
        ) : null}
      </View>

      {c.status.state === 'error' && c.status.last_error ? <Txt style={{ color: C.warn }}>{c.status.last_error}</Txt> : null}

      <View style={{ gap: S.sm }}>
        <Txt variant="label">Hazards right now</Txt>
        {factors.length ? (
          <View style={styles.chips}>
            {factors.map((f) => (
              <FactorChip key={f} id={f} showMuted />
            ))}
          </View>
        ) : (
          <Txt variant="muted">{running ? 'Nothing flagged at the moment.' : '—'}</Txt>
        )}
        <Txt variant="small">Circles on the live picture show what the model is looking at.</Txt>
      </View>

      <Txt variant="label" style={{ marginTop: S.sm }}>
        Recent alerts from this camera
      </Txt>
    </View>
  );

  return (
    <View style={{ flex: 1, backgroundColor: C.bg }}>
      <Stack.Screen options={{ title: c.name || name || 'Live view' }} />
      <FlatList<EventSummary>
        data={events.data?.items ?? []}
        keyExtractor={(e) => e.event_id}
        ListHeaderComponent={header}
        contentContainerStyle={styles.list}
        ItemSeparatorComponent={() => <View style={{ height: S.sm }} />}
        ListEmptyComponent={
          events.loading ? (
            <ActivityIndicator color={C.accent} style={{ marginTop: S.lg }} />
          ) : (
            <EmptyState icon="shield-checkmark-outline" title="No alerts yet" body="Flagged moments from this camera will appear here." />
          )
        }
        renderItem={({ item }) => (
          <EventRow
            ev={item}
            base={api.baseUrl}
            cameraName={c.name}
            onPress={(e) => router.push({ pathname: '/event/[id]', params: { id: e.event_id } })}
          />
        )}
      />
      <CameraOptionsSheet
        cam={menuOpen ? c : null}
        canControl={!!canControl}
        canDelete={isAdmin}
        busy={busy}
        onClose={() => setMenuOpen(false)}
        onToggle={async () => {
          await toggle();
        }}
        onDeleted={remove}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  list: { padding: S.lg, paddingBottom: S.xl * 2, gap: 0, maxWidth: 900, width: '100%', alignSelf: 'center' },
  video: { aspectRatio: 16 / 9, backgroundColor: '#000', borderRadius: R.lg, overflow: 'hidden', borderWidth: 1, borderColor: C.border },
  center: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center', padding: S.lg },
  liveTag: {
    position: 'absolute',
    top: S.sm,
    left: S.sm,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: R.sm,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  liveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: C.danger },
  liveText: { color: '#fff', fontSize: 11, fontFamily: 'IBMPlexSans_500Medium', letterSpacing: 0.4 },
  statusRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  state: { flexDirection: 'row', alignItems: 'center', gap: S.sm },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: S.sm },
});
