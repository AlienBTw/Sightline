import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from './api';

/** Small data-loading hook: loading / error / data with reload, and no state updates after unmount. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const alive = useRef(true);
  const seq = useRef(0);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const run = useCallback(
    async (mode: 'initial' | 'refresh' | 'silent' = 'initial') => {
      const my = ++seq.current;
      if (mode === 'initial') setLoading(true);
      if (mode === 'refresh') setRefreshing(true);
      try {
        const d = await fn();
        if (alive.current && my === seq.current) {
          setData(d);
          setError(null);
        }
      } catch (e) {
        if (alive.current && my === seq.current && !(e instanceof ApiError && e.status === 401)) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (alive.current && my === seq.current) {
          setLoading(false);
          setRefreshing(false);
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps, react-hooks/use-memo -- caller supplies deps
    deps
  );

  useEffect(() => {
    void run('initial');
  }, [run]);

  return { data, setData, error, loading, refreshing, reload: run };
}

export function errMsg(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}
