// Draws normalised hazard circles over a video frame (client-side overlay for clip.mp4).
import { StyleSheet, Text, View } from 'react-native';
import { factorInfo } from '../lib/factors';
import { type ActiveCircle, circleBox } from '../lib/overlay';
import { C } from '../lib/theme';

export function CircleOverlay({ circles, width, height }: { circles: ActiveCircle[]; width: number; height: number }) {
  if (!width || !height || !circles.length) return null;
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      {circles.map((c) => {
        const b = circleBox(c, width, height);
        const r = Math.max(b.width, b.height) / 2;
        return (
          <View key={c.factor} style={{ position: 'absolute', left: b.left, top: b.top, width: b.width, height: b.height }}>
            <View
              style={[
                styles.ring,
                { width: b.width, height: b.height, borderRadius: r, transform: [{ rotate: b.rotate }] },
              ]}
            />
            <View style={[styles.tag, { top: b.height + 2 }]}>
              <Text style={styles.tagText} numberOfLines={2}>
                {factorInfo(c.factor).name}
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  ring: {
    borderWidth: 3,
    borderColor: C.danger,
    backgroundColor: 'rgba(255,77,79,0.10)',
    shadowColor: '#000',
    elevation: 2,
  },
  tag: {
    position: 'absolute',
    left: 0,
    maxWidth: 160,
    backgroundColor: C.danger,
    borderRadius: 4,
    paddingHorizontal: 4,
    paddingVertical: 1,
  },
  tagText: { color: '#fff', fontSize: 11, fontWeight: '800' },
});
