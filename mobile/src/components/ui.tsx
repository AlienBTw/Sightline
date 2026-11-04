import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
  type PressableProps,
  type StyleProp,
  type TextStyle,
  type ViewStyle,
} from 'react-native';
import { factorInfo, riskLevel, isExperimental } from '../lib/factors';
import { pct } from '../lib/format';
import { C, FONT, R, RISK_COLOR, S, STATUS_COLOR, STATUS_LABEL } from '../lib/theme';

export type IconName = React.ComponentProps<typeof Ionicons>['name'];

export function Txt({
  style,
  variant = 'body',
  ...p
}: React.ComponentProps<typeof Text> & { variant?: 'title' | 'h2' | 'body' | 'muted' | 'small' | 'label' }) {
  return <Text {...p} style={[styles[variant], style as StyleProp<TextStyle>]} />;
}

type Kind = 'primary' | 'secondary' | 'danger' | 'ghost';

export function Button({
  title,
  icon,
  kind = 'primary',
  loading,
  disabled,
  style,
  compact,
  ...p
}: PressableProps & {
  title: string;
  icon?: IconName;
  kind?: Kind;
  loading?: boolean;
  compact?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  const bg = { primary: C.accent, secondary: C.surface, danger: C.dangerSoft, ghost: 'transparent' }[kind];
  const fg = { primary: C.accentText, secondary: C.text, danger: C.danger, ghost: C.accent }[kind];
  const border = { primary: C.accent, secondary: C.border, danger: C.danger, ghost: 'transparent' }[kind];
  const off = disabled || loading;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={title}
      disabled={off}
      android_ripple={{ color: 'rgba(255,255,255,0.12)' }}
      style={({ pressed }) => [
        styles.btn,
        compact && styles.btnCompact,
        { backgroundColor: bg, borderColor: border, opacity: off ? 0.5 : pressed ? 0.85 : 1 },
        style,
      ]}
      {...p}
    >
      {loading ? (
        <ActivityIndicator color={fg} size="small" />
      ) : (
        <>
          {icon ? <Ionicons name={icon} size={compact ? 16 : 18} color={fg} /> : null}
          <Text style={[styles.btnText, compact && { fontSize: 13 }, { color: fg }]} numberOfLines={1}>
            {title}
          </Text>
        </>
      )}
    </Pressable>
  );
}

export function IconButton({
  icon,
  label,
  onPress,
  color = C.text,
  size = 22,
}: {
  icon: IconName;
  label: string;
  onPress: () => void;
  color?: string;
  size?: number;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      hitSlop={8}
      onPress={onPress}
      style={({ pressed }) => [
        {
          minWidth: 48,
          minHeight: 48,
          alignItems: 'center',
          justifyContent: 'center',
          opacity: pressed ? 0.6 : 1,
        },
      ]}
    >
      <Ionicons name={icon} size={size} color={color} />
    </Pressable>
  );
}

export function Card({ style, children }: { style?: StyleProp<ViewStyle>; children: React.ReactNode }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function Pill({ label, color, solid, icon }: { label: string; color: string; solid?: boolean; icon?: IconName }) {
  return (
    <View
      style={[
        styles.pill,
        solid
          ? { borderColor: color, backgroundColor: color }
          : { borderLeftColor: color, backgroundColor: C.surface, borderColor: C.border },
      ]}
    >
      {icon ? <Ionicons name={icon} size={12} color={solid ? C.onSolid : color} /> : null}
      <Text style={[styles.pillText, { color: solid ? C.onSolid : C.text }]} numberOfLines={1}>
        {label}
      </Text>
    </View>
  );
}

export function StatusDot({ color }: { color: string }) {
  return <View style={[styles.dot, { backgroundColor: color }]} />;
}

export function RiskBadge({ risk, large }: { risk?: number | null; large?: boolean }) {
  const color = RISK_COLOR[riskLevel(risk)];
  return (
    <View style={[styles.risk, large && styles.riskLarge, { backgroundColor: color }]}>
      <Text style={[styles.riskText, large && { fontSize: 18 }]}>{pct(risk)}</Text>
      {large ? <Text style={styles.riskSub}>risk</Text> : null}
    </View>
  );
}

export function StatusPill({ status }: { status: string }) {
  return <Pill label={STATUS_LABEL[status] ?? status} color={STATUS_COLOR[status] ?? C.muted} />;
}

export function FactorChip({ id, active, onPress, showMuted }: { id: string; active?: boolean; onPress?: () => void; showMuted?: boolean }) {
  const f = factorInfo(id);
  const body = (
    <View style={[styles.chip, active && styles.chipActive]}>
      <Text style={[styles.chipText, active && { color: C.accentText }]} numberOfLines={1}>
        {f.name}
      </Text>
      {isExperimental(f.id) ? (
        <Text style={[styles.chipExp, active && { color: C.accentText }]} accessibilityLabel="experimental">
          exp.
        </Text>
      ) : null}
      {showMuted && f.muted ? <Ionicons name="notifications-off-outline" size={12} color={active ? C.accentText : C.muted} /> : null}
    </View>
  );
  if (!onPress) return body;
  return (
    <Pressable accessibilityRole="button" accessibilityLabel={`Filter ${f.name}`} onPress={onPress}>
      {body}
    </Pressable>
  );
}

export function FilterChip({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" accessibilityState={{ selected: active }} onPress={onPress}>
      <View style={[styles.chip, active && styles.chipActive]}>
        <Text style={[styles.chipText, active && { color: C.accentText, fontWeight: '700' }]}>{label}</Text>
      </View>
    </Pressable>
  );
}

export function Loading({ label }: { label?: string }) {
  return (
    <View style={styles.center}>
      <ActivityIndicator color={C.accent} size="large" />
      {label ? <Txt variant="muted" style={{ marginTop: S.md }}>{label}</Txt> : null}
    </View>
  );
}

export function EmptyState({ icon, title, body, action }: { icon: IconName; title: string; body?: string; action?: React.ReactNode }) {
  return (
    <View style={styles.center}>
      <Ionicons name={icon} size={44} color={C.faint} />
      <Txt variant="h2" style={{ marginTop: S.md, textAlign: 'center' }}>{title}</Txt>
      {body ? <Txt variant="muted" style={{ marginTop: S.xs, textAlign: 'center', maxWidth: 300 }}>{body}</Txt> : null}
      {action ? <View style={{ marginTop: S.lg }}>{action}</View> : null}
    </View>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <EmptyState
      icon="cloud-offline-outline"
      title="Something went wrong"
      body={message}
      action={onRetry ? <Button title="Try again" icon="refresh" kind="secondary" onPress={onRetry} /> : undefined}
    />
  );
}

export function Banner({ kind = 'info', text, icon }: { kind?: 'info' | 'warn' | 'error'; text: string; icon?: IconName }) {
  const color = { info: C.info, warn: C.warn, error: C.danger }[kind];
  const bg = { info: C.infoSoft, warn: C.warnSoft, error: C.dangerSoft }[kind];
  return (
    <View style={[styles.banner, { borderColor: color, backgroundColor: bg }]}>
      <Ionicons name={icon ?? (kind === 'error' ? 'alert-circle' : kind === 'warn' ? 'warning' : 'information-circle')} size={18} color={color} />
      <Text style={[styles.bannerText]}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  title: { color: C.text, fontFamily: FONT.sansSem, fontSize: 24, letterSpacing: -0.3 },
  h2: { color: C.text, fontFamily: FONT.sansSem, fontSize: 17 },
  body: { color: C.text, fontFamily: FONT.sans, fontSize: 15, lineHeight: 22 },
  muted: { color: C.muted, fontFamily: FONT.sans, fontSize: 14, lineHeight: 20 },
  small: { color: C.muted, fontFamily: FONT.sans, fontSize: 12 },
  label: { color: C.muted, fontFamily: FONT.sansMed, fontSize: 12, letterSpacing: 0.4 },
  btn: {
    minHeight: 48,
    paddingHorizontal: S.md,
    borderRadius: R.sm,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: S.sm,
    overflow: 'hidden',
  },
  btnCompact: { minHeight: 48, paddingHorizontal: S.md },
  btnText: { fontSize: 15, fontFamily: FONT.sansMed },
  card: { backgroundColor: C.surface, borderRadius: R.sm, borderWidth: 1, borderColor: C.border, overflow: 'hidden' },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    borderWidth: 1,
    borderRadius: R.sm,
    paddingHorizontal: 8,
    paddingVertical: 3,
    alignSelf: 'flex-start',
  },
  pillText: { fontSize: 12, fontFamily: FONT.sansMed },
  dot: { width: 8, height: 8, borderRadius: 4 },
  risk: { borderRadius: R.sm, paddingHorizontal: 8, paddingVertical: 3, minWidth: 48, alignItems: 'center' },
  riskLarge: { paddingHorizontal: 12, paddingVertical: 6, minWidth: 72 },
  riskText: { color: C.onSolid, fontFamily: FONT.mono, fontSize: 13 },
  riskSub: { color: C.onSolid, fontFamily: FONT.sans, fontSize: 11, marginTop: -2 },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    borderRadius: R.sm,
    borderWidth: 1,
    borderColor: C.border,
    backgroundColor: C.surface,
    paddingHorizontal: 12,
    paddingVertical: 10,
    minHeight: 48,
  },
  chipActive: { backgroundColor: C.accent, borderColor: C.accent },
  chipText: { color: C.text, fontFamily: FONT.sans, fontSize: 13 },
  chipExp: { color: C.muted, fontFamily: FONT.sans, fontSize: 11 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: S.xl },
  banner: { flexDirection: 'row', gap: S.sm, alignItems: 'flex-start', borderWidth: 1, borderRadius: R.sm, padding: S.md },
  bannerText: { color: C.text, fontFamily: FONT.sans, flex: 1, fontSize: 13, lineHeight: 18 },
});
