import { router, useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import { Alert, FlatList, RefreshControl, StyleSheet, TextInput, View, useWindowDimensions } from 'react-native';
import { CameraCard } from '../../components/CameraCard';
import { CameraOptionsSheet } from '../../components/CameraOptionsSheet';
import { Sheet } from '../../components/Sheet';
import { Banner, Button, EmptyState, ErrorState, Loading, Txt } from '../../components/ui';
import { useRealtime, useRealtimeState } from '../../lib/realtime';
import { useApi, useSession } from '../../lib/session';
import { C, R, S } from '../../lib/theme';
import type { Camera, CameraList } from '../../lib/types';
import { errMsg, useAsync } from '../../lib/useAsync';

export default function Cameras() {
  const api = useApi();
  const { user } = useSession();
  const isAdmin = user?.role === 'admin';
  const { cameraFactors } = useRealtimeState();
  const { width } = useWindowDimensions();
  const cols = width >= 900 ? 3 : width >= 600 ? 2 : 1;
  const list = useAsync<CameraList>(() => api.cameras(), [api]);
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [menuCam, setMenuCam] = useState<Camera | null>(null);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Refetch on focus (fresh signed snapshot links; one snapshot request per visit, never polled).
  const first = useRef(true);
  useFocusEffect(
    useCallback(() => {
      if (first.current) {
        first.current = false;
        return;
      }
      void list.reload('silent');
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])
  );

  useRealtime((m) => {
    if (m.type === 'camera' || m.type === 'camera_added' || m.type === 'camera_removed') {
      if (debounce.current) clearTimeout(debounce.current);
      debounce.current = setTimeout(() => void list.reload('silent'), 900);
    }
  });

  const toggle = useCallback(
    async (cam: Camera) => {
      const running = !['stopped', 'error', 'finished'].includes(cam.status.state);
      setBusy((b) => ({ ...b, [cam.id]: true }));
      try {
        const updated = running ? await api.stopCamera(cam.id) : await api.startCamera(cam.id);
        list.setData((d) => (d ? { ...d, items: d.items.map((c) => (c.id === cam.id ? updated : c)) } : d));
        setMenuCam((cur) => (cur?.id === cam.id ? updated : cur));
        setTimeout(() => void list.reload('silent'), running ? 500 : 9000); // model load takes ~8 s
      } catch (e) {
        Alert.alert(running ? 'Could not stop the camera' : 'Could not start the camera', errMsg(e));
      } finally {
        setBusy((b) => ({ ...b, [cam.id]: false }));
      }
    },
    [api, list]
  );

  const remove = useCallback(
    async (cam: Camera) => {
      try {
        await api.deleteCamera(cam.id);
        setMenuCam(null);
        void list.reload('silent');
      } catch (e) {
        Alert.alert('Could not delete', errMsg(e));
      }
    },
    [api, list]
  );

  // ------------------------------------------------------------ add RTSP (admin)
  const [rtspOpen, setRtspOpen] = useState(false);
  const [rtspName, setRtspName] = useState('');
  const [rtspUri, setRtspUri] = useState('');
  const [rtspBusy, setRtspBusy] = useState(false);
  const [rtspErr, setRtspErr] = useState<string | null>(null);

  async function addRtsp() {
    setRtspBusy(true);
    setRtspErr(null);
    try {
      await api.addCamera(rtspName.trim(), rtspUri.trim(), true);
      setRtspOpen(false);
      setRtspName('');
      setRtspUri('');
      void list.reload('silent');
    } catch (e) {
      setRtspErr(errMsg(e));
    } finally {
      setRtspBusy(false);
    }
  }

  if (list.loading && !list.data) return <Loading label="Loading cameras…" />;
  if (list.error && !list.data) return <ErrorState message={list.error} onRetry={() => void list.reload()} />;

  const items = list.data?.items ?? [];
  const cap = list.data?.capacity;
  const menuCanControl = !!menuCam && (isAdmin || menuCam.kind === 'virtual');

  const header = (
    <View style={{ gap: S.md, marginBottom: S.md }}>
      {isAdmin ? (
        <View style={styles.actions}>
          <Button title="Add RTSP" icon="add" onPress={() => setRtspOpen(true)} />
        </View>
      ) : null}
      {cap ? (
        <Txt variant="small">
          {cap.active} of {cap.max} analysis slots in use · {items.length} camera{items.length === 1 ? '' : 's'}
        </Txt>
      ) : null}
      {items.length ? <Txt variant="small">Hold a camera for options.</Txt> : null}
      {list.error ? <Banner kind="warn" text={list.error} /> : null}
    </View>
  );

  return (
    <View style={{ flex: 1, backgroundColor: C.bg }}>
      <FlatList
        key={cols}
        data={items}
        numColumns={cols}
        keyExtractor={(c) => c.id}
        contentContainerStyle={styles.list}
        columnWrapperStyle={cols > 1 ? { gap: S.md } : undefined}
        ItemSeparatorComponent={() => <View style={{ height: S.md }} />}
        ListHeaderComponent={header}
        refreshControl={
          <RefreshControl refreshing={list.refreshing} onRefresh={() => void list.reload('refresh')} tintColor={C.accent} colors={[C.accent]} progressBackgroundColor={C.surface} />
        }
        ListEmptyComponent={
          <EmptyState
            icon="videocam-off-outline"
            title="No cameras yet"
            body="Nothing is connected yet. An admin adds an RTSP camera."
          />
        }
        renderItem={({ item }) => (
          <CameraCard
            cam={item}
            base={api.baseUrl}
            liveFactors={cameraFactors[item.id]}
            canControl={isAdmin || item.kind === 'virtual'}
            busy={!!busy[item.id]}
            onOpen={(c) => router.push({ pathname: '/camera/[id]', params: { id: c.id, name: c.name } })}
            onToggle={toggle}
            onLongPress={setMenuCam}
          />
        )}
        initialNumToRender={4}
        windowSize={5}
        removeClippedSubviews
      />

      <CameraOptionsSheet
        cam={menuCam}
        canControl={menuCanControl}
        canDelete={isAdmin}
        busy={!!(menuCam && busy[menuCam.id])}
        onClose={() => setMenuCam(null)}
        onToggle={toggle}
        onDeleted={remove}
      />

      <Sheet visible={rtspOpen} title="Add an RTSP camera" onClose={() => !rtspBusy && setRtspOpen(false)}>
        <Txt variant="label">Name</Txt>
        <TextInput value={rtspName} onChangeText={setRtspName} maxLength={60} placeholder="Saw bench" placeholderTextColor={C.faint} style={styles.input} />
        <Txt variant="label">Stream URL</Txt>
        <TextInput
          value={rtspUri}
          onChangeText={setRtspUri}
          placeholder="rtsp://user:pass@192.168.1.20:8554/live"
          placeholderTextColor={C.faint}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          style={styles.input}
        />
        <Txt variant="small">Credentials stay on the server; the app only ever shows a redacted address.</Txt>
        {rtspErr ? <Banner kind="error" text={rtspErr} /> : null}
        <Button title="Add and start" icon="add" onPress={addRtsp} loading={rtspBusy} disabled={!rtspName.trim() || rtspUri.trim().length < 8} />
      </Sheet>
    </View>
  );
}

const styles = StyleSheet.create({
  list: { padding: S.lg, paddingBottom: S.xl * 2, maxWidth: 1200, width: '100%', alignSelf: 'center' },
  actions: { flexDirection: 'row', gap: S.md },
  input: {
    color: C.text,
    fontSize: 16,
    backgroundColor: C.bg,
    borderColor: C.border,
    borderWidth: 1,
    borderRadius: R.md,
    paddingHorizontal: S.md,
    minHeight: 48,
  },
});
