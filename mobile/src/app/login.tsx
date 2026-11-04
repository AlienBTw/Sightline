import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import React, { useEffect, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Banner, Button, Txt } from '../components/ui';
import { ApiError } from '../lib/api';
import { useSession } from '../lib/session';
import { prefs } from '../lib/storage';
import { C, FONT, R, S } from '../lib/theme';
import { ServerUrlError } from '../lib/url';

const logo = require('../../assets/splash-icon.png');

export default function Login() {
  const { signIn, server: savedServer, phase, retry, expiredReason } = useSession();
  const [server, setServer] = useState(savedServer);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- adopt the saved server once it loads
    setServer((s) => s || savedServer);
  }, [savedServer]);
  useEffect(() => {
    void prefs.getUsername().then((u) => u && setUsername((cur) => cur || u));
  }, []);

  const canSubmit = server.trim() && username.trim() && password && !busy;

  async function submit() {
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    try {
      await signIn(server, username, password);
      setPassword('');
    } catch (e) {
      if (e instanceof ServerUrlError) setError(e.message);
      else if (e instanceof ApiError) setError(e.message);
      else setError('Could not sign in. Check the server address.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          <View style={styles.brand}>
            <Image source={logo} style={styles.logo} contentFit="contain" />
            <Text style={styles.name}>Sightline</Text>
            <Txt variant="muted" style={{ textAlign: 'center' }}>
              Live workshop hazard watcher
            </Txt>
          </View>

          {phase === 'offline' ? (
            <View style={{ gap: S.sm, marginBottom: S.lg }}>
              <Banner kind="warn" text="Can't reach the server right now. Your session is kept on this phone." />
              <Button title="Reconnect" icon="refresh" kind="secondary" onPress={() => void retry()} />
            </View>
          ) : expiredReason ? (
            <View style={{ marginBottom: S.lg }}>
              <Banner kind="info" text="Your session ended. Please sign in again." />
            </View>
          ) : null}

          <View style={styles.form}>
            <Field label="Server">
              <TextInput
                value={server}
                onChangeText={setServer}
                placeholder="https://your-name.ngrok-free.app"
                placeholderTextColor={C.faint}
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                textContentType="URL"
                style={styles.input}
                returnKeyType="next"
              />
            </Field>
            <Field label="Username">
              <TextInput
                value={username}
                onChangeText={setUsername}
                placeholder="judge"
                placeholderTextColor={C.faint}
                autoCapitalize="none"
                autoCorrect={false}
                autoComplete="username"
                textContentType="username"
                style={styles.input}
                returnKeyType="next"
              />
            </Field>
            <Field label="Password">
              <View style={styles.pwRow}>
                <TextInput
                  value={password}
                  onChangeText={setPassword}
                  placeholder="••••••••"
                  placeholderTextColor={C.faint}
                  secureTextEntry={!showPw}
                  autoCapitalize="none"
                  autoCorrect={false}
                  autoComplete="password"
                  textContentType="password"
                  style={[styles.input, { flex: 1, borderWidth: 0 }]}
                  returnKeyType="go"
                  onSubmitEditing={submit}
                />
                <Pressable
                  onPress={() => setShowPw((v) => !v)}
                  accessibilityRole="button"
                  accessibilityLabel={showPw ? 'Hide password' : 'Show password'}
                  hitSlop={8}
                  style={{ minWidth: 48, minHeight: 48, alignItems: 'center', justifyContent: 'center' }}
                >
                  <Ionicons name={showPw ? 'eye-off-outline' : 'eye-outline'} size={20} color={C.muted} />
                </Pressable>
              </View>
            </Field>

            {error ? <Banner kind="error" text={error} /> : null}

            <Button title="Sign in" icon="log-in-outline" onPress={submit} loading={busy} disabled={!canSubmit} />
            <Txt variant="small" style={{ textAlign: 'center', marginTop: S.sm }}>
              Use the https address and demo login from the submission notes.
            </Txt>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <View style={{ gap: 6 }}>
      <Txt variant="label">{label}</Txt>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  scroll: { flexGrow: 1, justifyContent: 'center', padding: S.xl, maxWidth: 520, width: '100%', alignSelf: 'center' },
  brand: { alignItems: 'center', marginBottom: S.xl * 1.5, gap: 4 },
  logo: { width: 84, height: 84, marginBottom: S.sm },
  name: { color: C.text, fontFamily: FONT.sansSem, fontSize: 32, letterSpacing: -0.4 },
  form: { gap: S.lg },
  input: {
    color: C.text,
    fontFamily: FONT.sans,
    fontSize: 16,
    backgroundColor: C.surface,
    borderColor: C.border,
    borderWidth: 1,
    borderRadius: R.sm,
    paddingHorizontal: S.md,
    minHeight: 48,
  },
  pwRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.surface,
    borderColor: C.border,
    borderWidth: 1,
    borderRadius: R.sm,
  },
});
