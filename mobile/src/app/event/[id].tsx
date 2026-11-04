import { Ionicons } from '@expo/vector-icons';
import { Stack, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from 'react-native';
import { ClipPlayer } from '../../components/ClipPlayer';
import { Button, Card, ErrorState, FactorChip, Loading, RiskBadge, StatusPill, Txt } from '../../components/ui';
import { EXPERIMENTAL_NOTE, factorInfo, headlineFactor, isExperimental, visibleFactors } from '../../lib/factors';
import { clock, timeAgo } from '../../lib/format';
import { useRealtime } from '../../lib/realtime';
import { useApi } from '../../lib/session';
import { C, FONT, R, S } from '../../lib/theme';
import type { EscalationCall, EventDetail, EventStatus } from '../../lib/types';
import { absUrl } from '../../lib/url';
import { errMsg, useAsync } from '../../lib/useAsync';

type Mode = 'annotated' | 'overlay';

export default function EventScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const api = useApi();
  const ev = useAsync<EventDetail>(() => api.event(id), [api, id]);
  const cams = useAsync(() => api.cameras(), [api]);
  const [mode, setMode] = useState<Mode | null>(null);
  const [acting, setActing] = useState<EventStatus | null>(null);
  const reloads = useRef(0);

  const d = ev.data;
  const circlesReady = !!d?.media.clip_circles.ready;

  // clip_circles.mp4 may still be rendering right after the alert: check again shortly.
  useEffect(() => {
    if (!d || circlesReady) return;
    const t = setTimeout(() => void ev.reload('silent'), 4000);
    return () => clearTimeout(t);
  }, [d, circlesReady, ev]);

  useRealtime((m) => {
    if (m.type === 'event_status' && m.event_id === id) ev.setData((cur) => (cur ? { ...cur, status: m.status } : cur));
  });

  const onPlayerError = useCallback(() => {
    // Signed media links last 15 min: fetch fresh ones (a few times at most).
    if (reloads.current < 3) {
      reloads.current += 1;
      setTimeout(() => void ev.reload('silent'), 800);
    }
  }, [ev]);

  async function act(status: EventStatus) {
    setActing(status);
    try {
      const r =
        status === 'acknowledged' ? await api.ack(id) : status === 'dismissed' ? await api.dismiss(id) : await api.setStatus(id, status);
      ev.setData((cur) => (cur ? { ...cur, status: r.status } : cur));
      void ev.reload('silent');
    } catch (e) {
      Alert.alert('Could not update the alert', errMsg(e));
    } finally {
      setActing(null);
    }
  }

  if (ev.loading && !d) return <Loading label="Loading alert…" />;
  if (ev.error && !d) return <ErrorState message={ev.error} onRetry={() => void ev.reload()} />;
  if (!d) return null;

  const head = headlineFactor(d.top_factor, d.factors);
  const factors = visibleFactors(d.factors);
  const camName = cams.data?.items.find((c) => c.id === d.camera_id)?.name ?? d.camera_id;
  const [cw, ch] = d.clip_size ?? [960, 540];
  const useMode: Mode = (mode ?? 'annotated') === 'annotated' && circlesReady ? 'annotated' : 'overlay';
  const clipUrl = useMode === 'annotated' ? d.media.clip_circles.url : d.media.clip.url;
  const clipReady = useMode === 'annotated' ? circlesReady : d.media.clip.ready;
  const calls = (d.escalation?.calls ?? []).filter((c) => !factorInfo(c.factor).hidden);
  const dur = d.ended_at ? Math.max(0, (Date.parse(d.ended_at) - Date.parse(d.started_at)) / 1000) : null;

  return (
    <>
      <Stack.Screen options={{ title: head ? factorInfo(head).name : 'Alert' }} />
      <ScrollView
        style={{ flex: 1, backgroundColor: C.bg }}
        contentContainerStyle={styles.scroll}
        refreshControl={
          <RefreshControl refreshing={ev.refreshing} onRefresh={() => void ev.reload('refresh')} tintColor={C.accent} colors={[C.accent]} progressBackgroundColor={C.surface} />
        }
      >
        {clipReady ? (
          <ClipPlayer
            key={clipUrl}
            url={absUrl(api.baseUrl, clipUrl)}
            aspect={cw / ch}
            overlays={d.overlays}
            drawCircles={useMode === 'overlay'}
            onError={onPlayerError}
          />
        ) : (
          <View style={[styles.pending, { aspectRatio: cw / ch }]}>
            <Txt variant="muted">The clip is still being saved…</Txt>
          </View>
        )}

        <View style={styles.seg}>
          {(['annotated', 'overlay'] as Mode[]).map((m) => {
            const disabled = m === 'annotated' && !circlesReady;
            const on = useMode === m;
            return (
              <Pressable
                key={m}
                disabled={disabled}
                onPress={() => setMode(m)}
                accessibilityRole="button"
                accessibilityState={{ selected: on, disabled }}
                style={[styles.segBtn, on && styles.segOn, disabled && { opacity: 0.4 }]}
              >
                <Text style={[styles.segText, on && { color: C.accentText }]}>
                  {m === 'annotated' ? 'Annotated clip' : 'Original + live circles'}
                </Text>
              </Pressable>
            );
          })}
        </View>

        <View style={styles.headRow}>
          <View style={{ flex: 1, gap: 4 }}>
            <Txt variant="title" style={{ fontSize: 21 }}>
              {head ? factorInfo(head).name : 'Hazard'}
            </Txt>
            <Txt variant="muted">
              {camName} · {timeAgo(d.started_at)}
            </Txt>
            <Txt variant="small">
              {clock(d.started_at)}
              {dur != null ? ` · ${dur.toFixed(0)} s` : ''}
            </Txt>
            <View style={{ marginTop: 4 }}>
              <StatusPill status={d.status} />
            </View>
          </View>
          <RiskBadge risk={d.risk_max} large />
        </View>

        <View style={styles.actions}>
          <Button
            title={d.status === 'acknowledged' ? 'Acknowledged' : 'Acknowledge'}
            icon="checkmark-circle"
            onPress={() => act('acknowledged')}
            loading={acting === 'acknowledged'}
            disabled={d.status === 'acknowledged' || !!acting}
            style={{ flex: 1 }}
          />
          <Button
            title={d.status === 'dismissed' ? 'Dismissed' : 'Dismiss'}
            icon="close-circle"
            kind="secondary"
            onPress={() => act('dismissed')}
            loading={acting === 'dismissed'}
            disabled={d.status === 'dismissed' || !!acting}
            style={{ flex: 1 }}
          />
        </View>
        {d.status !== 'open' ? (
          <Button title="Reopen" icon="refresh" kind="ghost" compact onPress={() => act('open')} loading={acting === 'open'} disabled={!!acting} />
        ) : null}

        <Card style={styles.section}>
          <Txt variant="label">What was flagged</Txt>
          {factors.map((f) => {
            const seg = d.segments?.find((s) => s.factor === f);
            return (
              <View key={f} style={{ gap: 4 }}>
                <FactorChip id={f} showMuted />
                <Txt variant="muted">{factorInfo(f).plain}</Txt>
                {seg ? (
                  <Txt variant="small">
                    Seen for {(seg.end_s - seg.start_s).toFixed(1)} s
                    {seg.peak_score != null && seg.threshold ? ` · peak ${(seg.peak_score / seg.threshold).toFixed(2)}× the alert threshold` : ''}
                  </Txt>
                ) : null}
              </View>
            );
          })}
          {factors.some(isExperimental) ? (
            <Txt variant="small" style={{ fontStyle: 'italic' }}>
              {EXPERIMENTAL_NOTE}
            </Txt>
          ) : null}
          {!factors.length ? <Txt variant="muted">No visible factors.</Txt> : null}
        </Card>

        <Card style={styles.section}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
            <Ionicons name="sparkles" size={14} color={C.accent} />
            <Txt variant="label">Second opinion (Meta SAM 3.1 + Muse Spark)</Txt>
          </View>
          {calls.length ? (
            calls.map((c, i) => <Escalation key={`${c.factor}-${i}`} call={c} />)
          ) : (
            <Txt variant="muted">Not needed: the model was confident about this one.</Txt>
          )}
        </Card>

        {d.actions?.length ? (
          <Card style={styles.section}>
            <Txt variant="label">History</Txt>
            {d.actions.map((a, i) => (
              <Txt key={i} variant="small">
                {clock(new Date(a.at * 1000).toISOString())} · {a.status.replace('_', ' ')} by {a.by}
              </Txt>
            ))}
          </Card>
        ) : null}
      </ScrollView>
    </>
  );
}

function samText(s?: { status: string }) {
  switch (s?.status) {
    case 'ok':
      return 'SAM 3.1 outlined the object and tightened the circle.';
    case 'no_match':
      return 'SAM 3.1 found no matching object; the model’s own circle is shown.';
    case 'skipped_no_key':
      return 'SAM 3.1 not run (no Meta API key on the server).';
    case 'error':
      return 'SAM 3.1 call failed; the model’s own circle is shown.';
    case undefined:
      return null;
    default:
      return `SAM 3.1: ${s?.status}`;
  }
}

function Escalation({ call }: { call: EscalationCall }) {
  const sam = samText(call['sam3.1']);
  const v = call.muse_spark;
  const reason = call.reason?.startsWith('high_severity')
    ? 'Checked because this hazard is high severity.'
    : call.reason?.startsWith('uncertain')
      ? 'Checked because the score was close to the threshold.'
      : null;
  return (
    <View style={styles.esc}>
      <FactorChip id={call.factor} />
      {reason ? <Txt variant="small">{reason}</Txt> : null}
      {v?.status === 'ok' ? (
        <View style={{ gap: 4 }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
            <Ionicons
              name={v.confirmed ? 'checkmark-circle' : v.confirmed === false ? 'close-circle' : 'help-circle'}
              size={18}
              color={v.confirmed ? C.danger : v.confirmed === false ? C.ok : C.muted}
            />
            <Txt style={{ fontWeight: '700' }}>
              Muse Spark: {v.confirmed ? 'confirmed' : v.confirmed === false ? 'not confirmed' : 'unsure'}
              {v.severity ? ` · severity ${v.severity}/5` : ''}
            </Txt>
          </View>
          {v.reason ? <Txt variant="muted">{v.reason}</Txt> : null}
          {v.action ? (
            <View style={styles.action}>
              <Ionicons name="construct-outline" size={14} color={C.accent} />
              <Txt style={{ flex: 1, fontSize: 14 }}>{v.action}</Txt>
            </View>
          ) : null}
        </View>
      ) : v ? (
        <Txt variant="small">
          {v.status === 'skipped_no_key' ? 'Muse Spark not run (no Meta API key on the server).' : `Muse Spark: ${v.status}`}
        </Txt>
      ) : null}
      {sam ? <Txt variant="small">{sam}</Txt> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { padding: S.lg, gap: S.lg, paddingBottom: S.xl * 2, maxWidth: 900, width: '100%', alignSelf: 'center' },
  pending: { width: '100%', backgroundColor: C.surface2, borderRadius: R.lg, alignItems: 'center', justifyContent: 'center' },
  seg: { flexDirection: 'row', backgroundColor: C.surface, borderRadius: R.md, borderWidth: 1, borderColor: C.border, padding: 3 },
  segBtn: { flex: 1, minHeight: 48, paddingVertical: 10, borderRadius: R.sm, alignItems: 'center', justifyContent: 'center' },
  segOn: { backgroundColor: C.accent },
  segText: { color: C.text, fontFamily: FONT.sansSem, fontSize: 13 },
  headRow: { flexDirection: 'row', alignItems: 'flex-start', gap: S.md },
  actions: { flexDirection: 'row', gap: S.md },
  section: { padding: S.lg, gap: S.md },
  esc: { gap: 6, paddingTop: S.xs },
  action: { flexDirection: 'row', gap: 6, alignItems: 'flex-start', backgroundColor: C.surface2, borderRadius: R.sm, padding: S.sm },
});
