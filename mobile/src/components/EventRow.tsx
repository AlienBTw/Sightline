import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import { memo, useEffect, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { factorInfo, headlineFactor, isExperimental, visibleFactors } from '../lib/factors';
import { timeAgo } from '../lib/format';
import type { EventSummary } from '../lib/types';
import { C, FONT, R, S } from '../lib/theme';
import { absUrl } from '../lib/url';
import { RiskBadge, StatusPill } from './ui';

function EventRowImpl({
  ev,
  base,
  cameraName,
  onPress,
}: {
  ev: EventSummary;
  base: string;
  cameraName?: string;
  onPress: (ev: EventSummary) => void;
}) {
  const head = headlineFactor(ev.top_factor, ev.factors);
  // The thumbnail is written a moment after the alert (503 until then): retry a few times.
  const [thumbTry, setThumbTry] = useState(0);
  const [thumbFailed, setThumbFailed] = useState(false);
  useEffect(() => {
    if (!thumbFailed || thumbTry >= 4) return;
    const t = setTimeout(() => {
      setThumbFailed(false);
      setThumbTry((n) => n + 1);
    }, 2500);
    return () => clearTimeout(t);
  }, [thumbFailed, thumbTry]);
  const others = visibleFactors(ev.factors).filter((f) => f !== head);
  return (
    <Pressable
      onPress={() => onPress(ev)}
      android_ripple={{ color: 'rgba(26,35,50,0.06)' }}
      accessibilityRole="button"
      accessibilityLabel={`Alert ${head ? factorInfo(head).name : ''}, ${cameraName ?? ev.camera_id}`}
      style={({ pressed }) => [styles.row, pressed && { opacity: 0.9 }]}
    >
      <View style={styles.thumbBox}>
        {ev.thumb_url ? (
          <Image
            key={thumbTry}
            source={{ uri: absUrl(base, ev.thumb_url), cacheKey: `thumb-${ev.event_id}-${thumbTry}` }}
            onError={() => setThumbFailed(true)}
            style={styles.thumb}
            contentFit="cover"
            transition={150}
            recyclingKey={ev.event_id}
          />
        ) : (
          <Ionicons name="videocam-outline" size={22} color={C.faint} />
        )}
        {ev.status === 'open' ? <View style={styles.newDot} /> : null}
      </View>
      <View style={styles.mid}>
        <Text style={styles.title} numberOfLines={1}>
          {head ? factorInfo(head).name : 'Hazard'}
          {isExperimental(head) ? <Text style={styles.exp}>  experimental</Text> : null}
        </Text>
        <Text style={styles.sub} numberOfLines={1}>
          {cameraName ?? ev.camera_id} · {timeAgo(ev.started_at)}
        </Text>
        <View style={styles.meta}>
          <StatusPill status={ev.status} />
          {others.length ? <Text style={styles.more}>+{others.map((id) => factorInfo(id).name).join(', ')}</Text> : null}
          {ev.escalated ? <Ionicons name="sparkles" size={13} color={C.accent} /> : null}
        </View>
      </View>
      <RiskBadge risk={ev.risk_max} />
    </Pressable>
  );
}

export const EventRow = memo(EventRowImpl);

const styles = StyleSheet.create({
  exp: { color: C.muted, fontFamily: FONT.sans, fontSize: 12 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: S.md,
    padding: S.md,
    minHeight: 72,
    backgroundColor: C.surface,
    borderRadius: R.sm,
    borderWidth: 1,
    borderColor: C.border,
  },
  thumbBox: {
    width: 96,
    height: 54,
    borderRadius: R.sm,
    backgroundColor: C.surface2,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  thumb: { width: '100%', height: '100%' },
  newDot: {
    position: 'absolute',
    top: 4,
    left: 4,
    width: 9,
    height: 9,
    borderRadius: 5,
    backgroundColor: C.danger,
    borderWidth: 1,
    borderColor: C.surface,
  },
  mid: { flex: 1, gap: 3 },
  title: { color: C.text, fontFamily: FONT.sansSem, fontSize: 15 },
  sub: { color: C.muted, fontFamily: FONT.sans, fontSize: 12 },
  meta: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 2 },
  more: { color: C.muted, fontSize: 12 },
});
