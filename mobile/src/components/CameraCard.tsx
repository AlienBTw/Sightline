import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import { memo } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { factorInfo, visibleFactors } from '../lib/factors';
import { duration } from '../lib/format';
import { C, CAMERA_STATE, FONT, R, S } from '../lib/theme';
import type { Camera } from '../lib/types';
import { absUrl } from '../lib/url';
import { Button, Pill, StatusDot } from './ui';

function CameraCardImpl({
  cam,
  base,
  liveFactors,
  canControl,
  busy,
  onOpen,
  onToggle,
  onLongPress,
}: {
  cam: Camera;
  base: string;
  liveFactors?: string[];
  canControl: boolean;
  busy: boolean;
  onOpen: (c: Camera) => void;
  onToggle: (c: Camera) => void;
  onLongPress?: (c: Camera) => void;
}) {
  const st = CAMERA_STATE[cam.status.state] ?? { label: cam.status.state, color: C.muted };
  const running = cam.status.state === 'running';
  const active = cam.status.state !== 'stopped' && cam.status.state !== 'error' && cam.status.state !== 'finished';
  const factors = visibleFactors(liveFactors ?? cam.status.active_factors);
  return (
    <Pressable
      onPress={() => onOpen(cam)}
      onLongPress={onLongPress ? () => onLongPress(cam) : undefined}
      delayLongPress={380}
      accessibilityRole="button"
      accessibilityLabel={`${cam.name}, ${st.label}`}
      accessibilityHint={onLongPress ? 'Hold for camera options' : undefined}
      style={({ pressed }) => [styles.card, pressed && { opacity: 0.92 }]}
    >
      <View style={styles.thumb}>
        {running ? (
          <Image
            source={{ uri: absUrl(base, cam.preview.snapshot_url) }}
            style={StyleSheet.absoluteFill}
            contentFit="cover"
            cachePolicy="none"
            transition={200}
          />
        ) : (
          <View style={styles.placeholder}>
            <Ionicons name={cam.kind === 'virtual' ? 'film-outline' : 'videocam-outline'} size={34} color={C.faint} />
            <Text style={styles.placeholderText}>{active ? 'Starting…' : 'Camera is stopped'}</Text>
          </View>
        )}
        <View style={styles.topRow}>
          <View style={styles.state}>
            <StatusDot color={st.color} />
            <Text style={styles.stateText}>{st.label}</Text>
          </View>
          {factors.length ? <Pill label={`${factors.length} hazard${factors.length > 1 ? 's' : ''}`} color={C.danger} solid icon="warning" /> : null}
        </View>
      </View>
      <View style={styles.body}>
        <View style={{ flex: 1, gap: 2 }}>
          <Text style={styles.name} numberOfLines={1}>
            {cam.name}
          </Text>
          <Text style={styles.meta} numberOfLines={1}>
            {cam.kind === 'virtual' ? `Sample video${cam.video?.duration_s ? ` · ${duration(cam.video.duration_s)} loop` : ''}` : cam.kind === 'rtsp' ? 'RTSP camera' : 'Network stream'}
            {running && cam.status.fps ? ` · ${cam.status.fps.toFixed(1)} fps` : ''}
          </Text>
          {factors.length ? (
            <Text style={styles.hazard} numberOfLines={1}>
              {factors.map((f) => factorInfo(f).name).join(' · ')}
            </Text>
          ) : null}
          {cam.status.state === 'error' && cam.status.last_error ? (
            <Text style={styles.err} numberOfLines={2}>
              {cam.status.last_error}
            </Text>
          ) : null}
        </View>
        {canControl ? (
          <Button
            compact
            kind={active ? 'secondary' : 'primary'}
            icon={active ? 'stop' : 'play'}
            title={active ? 'Stop' : 'Start'}
            loading={busy}
            onPress={() => onToggle(cam)}
          />
        ) : null}
      </View>
    </Pressable>
  );
}

export const CameraCard = memo(CameraCardImpl);

const styles = StyleSheet.create({
  card: { backgroundColor: C.surface, borderRadius: R.sm, borderWidth: 1, borderColor: C.border, overflow: 'hidden', flex: 1 },
  thumb: { aspectRatio: 16 / 9, backgroundColor: '#000' },
  placeholder: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: C.surface2 },
  placeholderText: { color: C.faint, fontSize: 13 },
  topRow: { position: 'absolute', top: S.sm, left: S.sm, right: S.sm, flexDirection: 'row', justifyContent: 'space-between' },
  state: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(26,35,50,0.72)',
    borderRadius: R.sm,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  stateText: { color: '#fff', fontFamily: FONT.sansMed, fontSize: 12 },
  body: { flexDirection: 'row', alignItems: 'center', gap: S.md, padding: S.md },
  name: { color: C.text, fontFamily: FONT.sansSem, fontSize: 16 },
  meta: { color: C.muted, fontFamily: FONT.sans, fontSize: 12 },
  hazard: { color: C.danger, fontFamily: FONT.sansMed, fontSize: 12 },
  err: { color: C.warn, fontSize: 12 },
});
