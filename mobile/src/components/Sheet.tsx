import React from 'react';
import { KeyboardAvoidingView, Modal, Pressable, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { C, R, S } from '../lib/theme';
import { Txt } from './ui';

export function Sheet({
  visible,
  title,
  onClose,
  children,
  dismissable = true,
}: {
  visible: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  dismissable?: boolean;
}) {
  const insets = useSafeAreaInsets();
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={() => {
        if (dismissable) onClose();
      }} statusBarTranslucent>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior="height">
        <Pressable style={styles.backdrop} onPress={() => {
            if (dismissable) onClose();
          }} accessibilityLabel="Close" />
        <View style={[styles.sheet, { paddingBottom: insets.bottom + S.lg }]}>
          <View style={styles.handle} />
          <Txt variant="h2" style={{ marginBottom: S.md }}>
            {title}
          </Txt>
          {children}
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: C.overlay },
  sheet: {
    backgroundColor: C.surface,
    borderTopLeftRadius: R.lg * 1.5,
    borderTopRightRadius: R.lg * 1.5,
    borderWidth: 1,
    borderColor: C.border,
    padding: S.xl,
    gap: S.md,
    maxWidth: 640,
    width: '100%',
    alignSelf: 'center',
  },
  handle: { alignSelf: 'center', width: 40, height: 4, borderRadius: 2, backgroundColor: C.border, marginBottom: S.sm },
});
