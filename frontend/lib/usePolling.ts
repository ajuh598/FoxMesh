"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/** Fetch now and then every `intervalMs` while `active` is true. */
export function usePolling<T>(fetcher: () => Promise<T>, intervalMs: number, active = true) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      setData(await fetcherRef.current());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    if (!active) return;
    const timer = setInterval(refresh, intervalMs);
    return () => clearInterval(timer);
  }, [refresh, intervalMs, active]);

  return { data, error, refresh };
}
