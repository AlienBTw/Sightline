// Slide-down banner for new alerts while the app is open (WebSocket-driven).
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { Animated, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { riskLevel } from '../lib/factors';
import { alertText, onInAppAlert } from '../lib/notifications';
import { C, FONT, R, RISK_COLOR, S } from '../lib/theme';
import type { EventSummary } from '../lib/types';

export function InAppAlert() {
  const insets = useSafeAreaInsets();
  const [ev, setEv] = useState<EventSummary | null>(null);
  const y = useRef(new Animated.Value(-160)).current;
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () =>
      onInAppAlert((e) => {
        setEv(e);
        Animated.spring(y, { toValue: 0, useNativeDriver: true, speed: 16, bounciness: 6 }).start();
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(hide, 6000);
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  function hide() {
    Animated.timing(y, { toValue: -160, duration: 200, useNativeDriver: true }).start(() => setEv(null));
  }

  if (!ev) return null;
  const { title, body } = alertText(ev);
  const color = RISK_COLOR[riskLevel(ev.risk_max)];
  return (
    <Animated.View style={[styles.wrap, { top: insets.top + 6, transform: [{ translateY: y }] }]}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={title}
        onPress={() => {
          hide();
          router.push({ pathname: '/event/[id]', params: { id: ev.event_id } });
        }}
        style={[styles.card, { borderLeftColor: color }]}
      >
        <Ionicons name="warning" size={22} color={color} />
        <View style={{ flex: 1 }}>
          <Text style={styles.title} numberOfLines={1}>
            {title}
          </Text>
          <Text style={styles.body} numberOfLines={1}>
            {body}
          </Text>
        </View>
        <Ionicons name="chevron-forward" size={18} color={C.muted} />
      </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  wrap: { position: 'absolute', left: S.md, right: S.md, zIndex: 100, elevation: 12 },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: S.md,
    backgroundColor: C.surface2,
    borderRadius: R.lg,
    borderWidth: 1,
    borderColor: C.border,
    borderLeftWidth: 5,
    padding: S.md,
  },
  title: { color: C.text, fontFamily: FONT.sansSem, fontSize: 15 },
  body: { color: C.muted, fontSize: 13 },
});
