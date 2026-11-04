// Plays an event clip with expo-video (range-streamed MP4). In "overlay" mode the circles are drawn
// client-side over the clean clip, synced to playback time from the event's overlays (t_ms).
import { useEvent } from 'expo';
import { useVideoPlayer, VideoView } from 'expo-video';
import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { factorInfo } from '../lib/factors';
import { circlesAt } from '../lib/overlay';
import { C, R } from '../lib/theme';
import type { Overlay } from '../lib/types';
import { CircleOverlay } from './CircleOverlay';
import { Txt } from './ui';

export function ClipPlayer({
  url,
  aspect,
  overlays,
  drawCircles,
  onError,
}: {
  url: string;
  aspect: number;
  overlays?: Overlay[];
  drawCircles: boolean;
  onError?: (msg: string) => void;
}) {
  const source = useMemo(() => ({ uri: url, headers: { 'ngrok-skip-browser-warning': '1' } }), [url]);
  const player = useVideoPlayer(source, (p) => {
    p.loop = true;
    p.muted = true;
    p.timeUpdateEventInterval = 0.1;
    p.play();
  });
  const { status, error } = useEvent(player, 'statusChange', { status: player.status });
  const [tMs, setTMs] = useState(0);
  const [box, setBox] = useState({ w: 0, h: 0 });

  useEffect(() => {
    if (!drawCircles) return;
    const sub = player.addListener('timeUpdate', ({ currentTime }) => setTMs(currentTime * 1000));
    return () => sub.remove();
  }, [player, drawCircles]);

  useEffect(() => {
    if (status === 'error') onError?.(error?.message ?? 'playback error');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const circles = useMemo(
    () => (drawCircles ? circlesAt(overlays, tMs, { hidden: (f) => !!factorInfo(f).hidden }) : []),
    [drawCircles, overlays, tMs]
  );

  return (
    <View style={[styles.box, { aspectRatio: aspect }]}>
      <View style={StyleSheet.absoluteFill} onLayout={(e) => setBox({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}>
        <VideoView
          player={player}
          style={StyleSheet.absoluteFill}
          contentFit="contain"
          nativeControls
          allowsPictureInPicture={false}
        />
        {drawCircles ? <CircleOverlay circles={circles} width={box.w} height={box.h} /> : null}
      </View>
      {status === 'loading' ? (
        <View style={styles.center} pointerEvents="none">
          <ActivityIndicator color={C.accent} />
        </View>
      ) : null}
      {status === 'error' ? (
        <View style={styles.center} pointerEvents="none">
          <Txt variant="muted">Could not play the clip. Reloading…</Txt>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  box: { width: '100%', backgroundColor: '#000', borderRadius: R.lg, overflow: 'hidden', borderWidth: 1, borderColor: C.border },
  center: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
});
