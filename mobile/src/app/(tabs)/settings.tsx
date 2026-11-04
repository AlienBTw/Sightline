import { Ionicons } from '@expo/vector-icons';
import Constants from 'expo-constants';
import React, { useState, useSyncExternalStore } from 'react';
import { Alert, Linking, ScrollView, StyleSheet, View } from 'react-native';
import { Button, Card, FactorChip, Pill, Txt } from '../../components/ui';
import { FACTORS, isExperimental } from '../../lib/factors';
import {
  getPushStatus,
  registerForPush,
  sendLocalTest,
  setPushStatus,
  subscribePushStatus,
  unregisterPush,
} from '../../lib/notifications';
import { useRealtimeState } from '../../lib/realtime';
import { useApi, useSession } from '../../lib/session';
import { C, S } from '../../lib/theme';
import { errMsg, useAsync } from '../../lib/useAsync';

export default function Settings() {
  const api = useApi();
  const { user, server, signOut } = useSession();
  const { connected } = useRealtimeState();
  const push = useSyncExternalStore(subscribePushStatus, getPushStatus);
  const health = useAsync(() => api.healthDetail(), [api]);
  const [busy, setBusy] = useState<string | null>(null);

  async function testPush() {
    setBusy('push');
    try {
      const r = await api.testPush();
      Alert.alert('Test push sent', `Sent to ${r.sent} device${r.sent === 1 ? '' : 's'}. It should arrive in a few seconds.`);
    } catch (e) {
      Alert.alert('Test push', errMsg(e));
    } finally {
      setBusy(null);
    }
  }

  async function retryPush() {
    setBusy('register');
    try {
      setPushStatus(await registerForPush(api));
    } finally {
      setBusy(null);
    }
  }

  async function localTest() {
    const ok = await sendLocalTest();
    if (!ok) Alert.alert('Notifications are off', 'Allow notifications for Sightline in Android settings.');
  }

  function confirmLogout() {
    Alert.alert('Sign out?', 'This phone will stop receiving alerts until you sign in again.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Sign out',
        style: 'destructive',
        onPress: async () => {
          await unregisterPush(api);
          setPushStatus(null);
          await signOut();
        },
      },
    ]);
  }

  const h = health.data as
    | { version?: string; escalation_key_configured?: boolean; capacity?: { active: number; max: number }; disk_free_gb?: number }
    | null;

  return (
    <ScrollView style={{ flex: 1, backgroundColor: C.bg }} contentContainerStyle={styles.scroll}>
      <Card style={styles.card}>
        <Txt variant="label">Account</Txt>
        <Row icon="person-circle-outline" label={user?.username ?? '—'} right={<Pill label={user?.role ?? ''} color={user?.role === 'admin' ? C.accent : C.info} />} />
        <Row icon="server-outline" label={server} />
        <Row
          icon="pulse-outline"
          label={connected ? 'Live updates connected' : 'Live updates reconnecting…'}
          right={<View style={[styles.dot, { backgroundColor: connected ? C.ok : C.warn }]} />}
        />
        {h ? (
          <>
            <Row icon="hardware-chip-outline" label={`Backend ${h.version ?? ''} · ${h.capacity?.active ?? 0}/${h.capacity?.max ?? 0} cameras analysing`} />
            <Row
              icon="sparkles-outline"
              label={h.escalation_key_configured ? 'Meta SAM 3.1 + Muse Spark escalation on' : 'Meta escalation off (no API key on server)'}
            />
          </>
        ) : health.error ? (
          <Txt variant="small">{health.error}</Txt>
        ) : null}
        <Button title="Sign out" icon="log-out-outline" kind="danger" onPress={confirmLogout} />
      </Card>

      <Card style={styles.card}>
        <Txt variant="label">Notifications</Txt>
        {push?.state === 'registered' ? (
          <Row icon="notifications" label="Push notifications are on for this phone." />
        ) : push ? (
          <Txt variant="muted">{push.reason} You still get an alert banner and a notification while Sightline is open or in the background.</Txt>
        ) : (
          <Txt variant="muted">Checking…</Txt>
        )}
        <View style={styles.btnRow}>
          {push?.state === 'registered' ? (
            <Button title="Send test push" icon="paper-plane-outline" onPress={testPush} loading={busy === 'push'} style={{ flex: 1 }} />
          ) : (
            <Button title="Retry push setup" icon="refresh" kind="secondary" onPress={retryPush} loading={busy === 'register'} style={{ flex: 1 }} />
          )}
          <Button title="Test on phone" icon="notifications-outline" kind="secondary" onPress={localTest} style={{ flex: 1 }} />
        </View>
        {push?.state === 'denied' ? <Button title="Open Android settings" kind="ghost" onPress={() => void Linking.openSettings()} /> : null}
      </Card>

      <Card style={styles.card}>
        <Txt variant="label">Hazards Sightline watches</Txt>
        {FACTORS.filter((f) => !f.hidden).map((f) => (
          <View key={f.id} style={{ gap: 2 }}>
            <FactorChip id={f.id} showMuted />
            <Txt variant="small" style={{ marginLeft: 4 }}>
              {f.plain}
              {f.muted ? ' Shown, but no alerts (too many false alarms in testing).' : ''}
              {isExperimental(f.id) ? ' Experimental.' : ' Verified in live testing.'}
            </Txt>
          </View>
        ))}
      </Card>

      <Txt variant="small" style={{ textAlign: 'center' }}>
        Sightline {Constants.expoConfig?.version ?? ''} · V-JEPA 2.1 + Meta SAM 3.1 + Muse Spark
      </Txt>
    </ScrollView>
  );
}

function Row({ icon, label, right }: { icon: React.ComponentProps<typeof Ionicons>['name']; label: string; right?: React.ReactNode }) {
  return (
    <View style={styles.row}>
      <Ionicons name={icon} size={20} color={C.muted} />
      <Txt style={{ flex: 1 }} numberOfLines={2}>
        {label}
      </Txt>
      {right}
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { padding: S.lg, gap: S.lg, paddingBottom: S.xl * 2, maxWidth: 700, width: '100%', alignSelf: 'center' },
  card: { padding: S.lg, gap: S.md },
  row: { flexDirection: 'row', alignItems: 'center', gap: S.md, minHeight: 48 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  btnRow: { flexDirection: 'row', gap: S.md },
});
