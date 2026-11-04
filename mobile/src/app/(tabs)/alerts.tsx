import { router } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, FlatList, RefreshControl, ScrollView, StyleSheet, View } from 'react-native';
import { EventRow } from '../../components/EventRow';
import { Banner, EmptyState, ErrorState, FilterChip, Loading, Txt } from '../../components/ui';
import { factorInfo, visibleFactors } from '../../lib/factors';
import { useRealtime } from '../../lib/realtime';
import { useApi } from '../../lib/session';
import { C, S } from '../../lib/theme';
import type { EventStatus, EventSummary } from '../../lib/types';
import { errMsg } from '../../lib/useAsync';

const STATUS_FILTERS: { key: string; label: string }[] = [
  { key: '', label: 'All' },
  { key: 'open', label: 'Open' },
  { key: 'acknowledged', label: 'Acknowledged' },
  { key: 'dismissed', label: 'Dismissed' },
];
const PAGE = 25;

function hazardIds(events: EventSummary[]): string[] {
  const ids: string[] = [];
  for (const ev of events) {
    for (const id of visibleFactors(ev.factors)) {
      if (!ids.includes(id)) ids.push(id);
    }
  }
  return ids;
}

export default function Alerts() {
  const api = useApi();
  const [status, setStatus] = useState('');
  const [factor, setFactor] = useState('');
  const [items, setItems] = useState<EventSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [next, setNext] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [more, setMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [names, setNames] = useState<Record<string, string>>({});
  const [hazards, setHazards] = useState<string[]>([]);
  const seq = useRef(0);
  const thumbTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = useCallback(
    async (mode: 'initial' | 'refresh' | 'silent') => {
      const my = ++seq.current;
      if (mode === 'initial') setLoading(true);
      if (mode === 'refresh') setRefreshing(true);
      try {
        const r = await api.events({ status: status || undefined, factor: factor || undefined, limit: PAGE, offset: 0 });
        if (my !== seq.current) return;
        setItems(r.items);
        setTotal(r.total);
        setNext(r.next_offset);
        setError(null);
        const catalog = factor
          ? await api.events({ status: status || undefined, limit: 100, offset: 0 })
          : r;
        if (my !== seq.current) return;
        const ids = hazardIds(catalog.items);
        setHazards(ids);
        if (factor && !ids.includes(factor)) {
          setFactor('');
          return;
        }
      } catch (e) {
        if (my === seq.current) setError(errMsg(e));
      } finally {
        if (my === seq.current) {
          setLoading(false);
          setRefreshing(false);
        }
      }
    },
    [api, status, factor]
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- async fetch on mount; state updates happen after awaits
    void load('initial');
  }, [load]);

  useEffect(() => {
    api
      .cameras()
      .then((r) => setNames(Object.fromEntries(r.items.map((c) => [c.id, c.name]))))
      .catch(() => {});
  }, [api]);

  async function loadMore() {
    if (next == null || more) return;
    setMore(true);
    const my = seq.current;
    try {
      const r = await api.events({ status: status || undefined, factor: factor || undefined, limit: PAGE, offset: next });
      if (my !== seq.current) return;
      setItems((cur) => {
        const seen = new Set(cur.map((e) => e.event_id));
        return [...cur, ...r.items.filter((e) => !seen.has(e.event_id))];
      });
      if (!factor) {
        setHazards((cur) => {
          const ids = [...cur];
          for (const id of hazardIds(r.items)) if (!ids.includes(id)) ids.push(id);
          return ids;
        });
      }
      setNext(r.next_offset);
      setTotal(r.total);
    } catch {
      // keep what we have; pull to refresh retries
    } finally {
      setMore(false);
    }
  }

  const matches = useCallback(
    (e: EventSummary) => (!status || e.status === status) && (!factor || e.factors.includes(factor)),
    [status, factor]
  );

  useRealtime((m) => {
    if (m.type === 'event') {
      const ev = { ...m.event, status: (m.event.status ?? 'open') as EventStatus };
      setHazards((cur) => {
        const ids = [...cur];
        for (const id of visibleFactors(ev.factors)) if (!ids.includes(id)) ids.push(id);
        return ids;
      });
      if (!matches(ev)) return;
      setItems((cur) => (cur.some((e) => e.event_id === ev.event_id) ? cur : [ev, ...cur]));
      setTotal((t) => t + 1);
      if (!names[ev.camera_id]) {
        api.cameras().then((r) => setNames(Object.fromEntries(r.items.map((c) => [c.id, c.name])))).catch(() => {});
      }
      // WebSocket summaries carry no signed thumbnail link: fetch the page once things settle.
      if (thumbTimer.current) clearTimeout(thumbTimer.current);
      thumbTimer.current = setTimeout(() => void load('silent'), 4000);
    } else if (m.type === 'event_status') {
      setItems((cur) =>
        cur
          .map((e) => (e.event_id === m.event_id ? { ...e, status: m.status } : e))
          .filter((e) => e.event_id !== m.event_id || matches(e))
      );
    }
  });

  const open = useCallback((e: EventSummary) => router.push({ pathname: '/event/[id]', params: { id: e.event_id } }), []);

  const header = (
    <View style={{ gap: S.sm, marginBottom: S.md }}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
        {STATUS_FILTERS.map((f) => (
          <FilterChip key={f.key} label={f.label} active={status === f.key} onPress={() => setStatus(f.key)} />
        ))}
      </ScrollView>
      {hazards.length ? (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips}>
          <FilterChip label="All hazards" active={!factor} onPress={() => setFactor('')} />
          {hazards.map((id) => (
            <FilterChip
              key={id}
              label={factorInfo(id).name}
              active={factor === id}
              onPress={() => setFactor(factor === id ? '' : id)}
            />
          ))}
        </ScrollView>
      ) : null}
      <Txt variant="small">
        {total} alert{total === 1 ? '' : 's'}
        {status ? ` · ${STATUS_FILTERS.find((s) => s.key === status)?.label.toLowerCase()}` : ''}
      </Txt>
      {error && items.length ? <Banner kind="warn" text={error} /> : null}
    </View>
  );

  if (loading && !items.length) return <Loading label="Loading alerts…" />;
  if (error && !items.length) return <ErrorState message={error} onRetry={() => void load('initial')} />;

  return (
    <FlatList
      style={{ flex: 1, backgroundColor: C.bg }}
      data={items}
      keyExtractor={(e) => e.event_id}
      contentContainerStyle={styles.list}
      ListHeaderComponent={header}
      ItemSeparatorComponent={() => <View style={{ height: S.sm }} />}
      renderItem={({ item }) => <EventRow ev={item} base={api.baseUrl} cameraName={names[item.camera_id]} onPress={open} />}
      onEndReached={loadMore}
      onEndReachedThreshold={0.4}
      ListFooterComponent={more ? <ActivityIndicator color={C.accent} style={{ margin: S.lg }} /> : undefined}
      ListEmptyComponent={
        <EmptyState
          icon="shield-checkmark-outline"
          title={status || factor ? 'No alerts match' : 'No alerts yet'}
          body={status || factor ? 'Try another filter.' : 'When a camera flags a hazard it shows up here instantly.'}
        />
      }
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => void load('refresh')} tintColor={C.accent} colors={[C.accent]} progressBackgroundColor={C.surface} />
      }
      initialNumToRender={10}
      windowSize={7}
      removeClippedSubviews
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: S.lg, paddingBottom: S.xl * 2, maxWidth: 900, width: '100%', alignSelf: 'center' },
  chips: { gap: S.sm, paddingRight: S.lg },
});
