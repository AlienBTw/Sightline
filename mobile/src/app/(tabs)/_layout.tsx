import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRealtimeState } from '../../lib/realtime';
import { C, FONT } from '../../lib/theme';

function LiveIndicator() {
  const { connected } = useRealtimeState();
  return (
    <View style={styles.live} accessibilityLabel={connected ? 'Live updates connected' : 'Live updates reconnecting'}>
      <View style={[styles.dot, { backgroundColor: connected ? C.ok : C.warn }]} />
      <Text style={styles.liveText}>{connected ? 'Live' : 'Reconnecting'}</Text>
    </View>
  );
}

export default function TabsLayout() {
  // Edge-to-edge: keep the tab bar above the gesture / 3-button navigation bar.
  const { bottom } = useSafeAreaInsets();
  return (
    <Tabs
      screenOptions={{
        headerStyle: { backgroundColor: C.surface },
        headerTintColor: C.text,
        headerTitleStyle: { fontFamily: FONT.sansSem, fontSize: 20, fontWeight: '600' },
        headerShadowVisible: false,
        headerRight: () => <LiveIndicator />,
        tabBarStyle: { backgroundColor: C.surface, borderTopColor: C.border, height: 58 + bottom, paddingBottom: 6 + bottom, paddingTop: 6 },
        tabBarActiveTintColor: C.accent,
        tabBarInactiveTintColor: C.muted,
        tabBarLabelStyle: { fontFamily: FONT.sansMed, fontSize: 12 },
        sceneStyle: { backgroundColor: C.bg },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Cameras',
          tabBarIcon: ({ color, size }) => <Ionicons name="videocam" size={size} color={color as string} />,
        }}
      />
      <Tabs.Screen
        name="alerts"
        options={{
          title: 'Alerts',
          tabBarIcon: ({ color, size }) => <Ionicons name="warning" size={size} color={color as string} />,
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color, size }) => <Ionicons name="settings-sharp" size={size} color={color as string} />,
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  live: { flexDirection: 'row', alignItems: 'center', gap: 6, marginRight: 16, minHeight: 48, paddingHorizontal: 4 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  liveText: { color: C.muted, fontFamily: FONT.sansMed, fontSize: 12 },
});
