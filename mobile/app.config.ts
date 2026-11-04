import type { ExpoConfig } from 'expo/config';

// Optional build-time default for the server URL (shown pre-filled on the login screen).
// It is NOT a secret: set SIGHTLINE_DEFAULT_SERVER=https://<name>.ngrok-free.app before building.
const defaultServerUrl = process.env.SIGHTLINE_DEFAULT_SERVER ?? '';

const config: ExpoConfig = {
  name: 'Sightline',
  slug: 'sightline',
  scheme: 'sightline',
  version: '1.0.0',
  orientation: 'portrait',
  icon: './assets/icon.png',
  userInterfaceStyle: 'light',
  backgroundColor: '#f4f6f8',
  platforms: ['android'],
  android: {
    package: 'com.alienbtw.sightline',
    versionCode: 1,
    adaptiveIcon: {
      backgroundColor: '#f4f6f8',
      foregroundImage: './assets/android-icon-foreground.png',
      backgroundImage: './assets/android-icon-background.png',
      monochromeImage: './assets/android-icon-monochrome.png',
    },
    // Minimal permissions: network + notifications only (no camera, storage or location).
    permissions: ['android.permission.INTERNET', 'android.permission.POST_NOTIFICATIONS'],
    blockedPermissions: [
      // Pulled in by libraries but unused: biometrics (expo-secure-store), boot-time rescheduling, the Play
      // install referrer and launcher badge counters (expo-notifications).
      'android.permission.USE_BIOMETRIC',
      'android.permission.USE_FINGERPRINT',
      'android.permission.RECEIVE_BOOT_COMPLETED',
      'com.google.android.finsky.permission.BIND_GET_INSTALL_REFERRER_SERVICE',
      'android.permission.READ_APP_BADGE',
      'com.sec.android.provider.badge.permission.READ',
      'com.sec.android.provider.badge.permission.WRITE',
      'com.htc.launcher.permission.READ_SETTINGS',
      'com.htc.launcher.permission.UPDATE_SHORTCUT',
      'com.sonyericsson.home.permission.BROADCAST_BADGE',
      'com.sonymobile.home.permission.PROVIDER_INSERT_BADGE',
      'com.anddoes.launcher.permission.UPDATE_COUNT',
      'com.majeur.launcher.permission.UPDATE_BADGE',
      'com.huawei.android.launcher.permission.CHANGE_BADGE',
      'com.huawei.android.launcher.permission.READ_SETTINGS',
      'com.huawei.android.launcher.permission.WRITE_SETTINGS',
      'com.oppo.launcher.permission.READ_SETTINGS',
      'com.oppo.launcher.permission.WRITE_SETTINGS',
      'me.everything.badger.permission.BADGE_COUNT_READ',
      'me.everything.badger.permission.BADGE_COUNT_WRITE',
      'android.permission.RECORD_AUDIO',
      'android.permission.READ_EXTERNAL_STORAGE',
      'android.permission.WRITE_EXTERNAL_STORAGE',
      'android.permission.SYSTEM_ALERT_WINDOW',
    ],
    allowBackup: false,
    predictiveBackGestureEnabled: false,
    ...(process.env.SIGHTLINE_GOOGLE_SERVICES ? { googleServicesFile: process.env.SIGHTLINE_GOOGLE_SERVICES } : {}),
  },
  plugins: [
    'expo-router',
    'expo-secure-store',
    ['expo-video', { supportsBackgroundPlayback: false, supportsPictureInPicture: false }],
    'expo-image',
    [
      'expo-splash-screen',
      { backgroundColor: '#f4f6f8', image: './assets/splash-icon.png', imageWidth: 160, resizeMode: 'contain' },
    ],
    [
      'expo-notifications',
      { icon: './assets/notification-icon.png', color: '#c45c26', defaultChannel: 'alerts' },
    ],
    [
      'expo-build-properties',
      { android: { enableMinifyInReleaseBuilds: true, enableShrinkResourcesInReleaseBuilds: true } },
    ],
    './plugins/withLoopbackCleartext',
    './plugins/withReleaseSigning',
  ],
  experiments: { typedRoutes: false },
  extra: {
    defaultServerUrl,
    ...(process.env.EAS_PROJECT_ID ? { eas: { projectId: process.env.EAS_PROJECT_ID } } : {}),
  },
};

export default config;
