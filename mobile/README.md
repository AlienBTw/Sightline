# Sightline mobile (Android)

The supervisor app for Sightline. It signs in to your Sightline backend, shows your cameras live with the hazard circles, alerts you in real time, and lets you review each flagged clip and acknowledge or dismiss it.

Built with Expo SDK 58 (React Native 0.88, TypeScript, expo-router). Android only.

## Screens

| Screen | What it does |
|---|---|
| **Sign in** | Server URL (your ngrok domain) plus username and password. The URL is set at runtime, so nothing about your server is baked into the app. |
| **Cameras** | A grid of cameras with a snapshot, a state badge and live hazard chips. You can start or stop a camera, upload a video (it becomes a looping virtual camera), and, as admin, add an RTSP camera or long-press to delete one. |
| **Live view** | The backend's low-bandwidth MJPEG stream (1 fps, 480 px) with the circles drawn on the server, live hazard chips, and that camera's recent alerts. A viewing session lasts about 5 minutes, then "Keep watching" resumes it, to save ngrok bandwidth. |
| **Alerts** | A real-time feed over WebSocket, with status and factor filters, risk colours, thumbnails and pagination. |
| **Event detail** | Plays the annotated clip (`clip_circles.mp4`), or the original clip with circles drawn by the app in sync with playback. Shows each factor in plain words with its time segments, the SAM 3.1 / Muse Spark escalation verdict, and the action history. You can acknowledge, dismiss or reopen it. |
| **Settings** | Account, server, connection state, backend health, notification status, a test notification, sign out, and a legend of the hazard factors. |

Tapping a notification opens the event, after sign-in if needed.

**Experimental factors:** in live testing only hand-in-danger-zone (F1) detections held up. Every other factor is marked *experimental* in chips, alert rows, notifications and the event page, and F1 wins the headline when an event has both. This is display only; the backend thresholds are unchanged.

## Security

- The access token lives only in memory. The refresh token is kept in the Android Keystore through `expo-secure-store` (this device only, not backed up). When the access token expires the app refreshes it once, shares that refresh between parallel requests, and signs out if the refresh is rejected.
- Tokens never go into URLs. The exceptions are the short-lived signed media links (`?st=`, 15 min) and the one-time WebSocket ticket (60 s), both issued by the backend.
- HTTPS only. Plain HTTP is allowed only for `127.0.0.1`, `localhost` and `10.0.2.2` (emulator and USB development), and Android's network security config enforces the same rule.
- No secrets in the app bundle. The server URL is entered at sign-in, and there is an optional build-time default (`SIGHTLINE_DEFAULT_SERVER`) that isn't a secret.
- The release keystore and its passwords live outside the repo (by default `F:\models\keys\`) and reach Gradle through environment variables.
- Minimal permissions: internet and notifications. Storage, microphone and overlay permissions are removed, and app backup is off.

## Setup

You need Node 20+ (tested with 24), the Android SDK (Android Studio), and a JDK (Android Studio's bundled JBR works with Gradle 9.4).

```powershell
cd mobile
npm install          # .npmrc sets legacy-peer-deps (React 19 peer ranges)
npm run typecheck
npm run lint
npm test             # unit tests: API client + token refresh, overlay timing, URL rules
```

## Development

The app uses native modules, so it runs as a development build rather than in Expo Go.

```powershell
npx expo run:android            # builds, installs on the emulator / USB phone, starts Metro
adb reverse tcp:8000 tcp:8000   # lets the device reach a local backend
```

Then sign in with `http://127.0.0.1:8000`. On the emulator, `http://10.0.2.2:8000` also works without `adb reverse`.

To exercise the API client against a running backend from Node (no phone needed):

```powershell
$env:SL_BASE="http://127.0.0.1:8000"; $env:SL_USER="..."; $env:SL_PASS="..."
$env:SL_ADMIN_USER="..."; $env:SL_ADMIN_PASS="..."     # optional: admin checks
npm run smoke -- ..\data\demo\sightline_demo.mp4       # optional video: upload checks
```

This covers sign-in, refresh rotation and reuse detection, auto-refresh on 401, events, signed media, ack/dismiss, the WebSocket ticket, uploads, the live view and logout. It creates a test camera and deletes it at the end (that needs the admin login).

## Building the APK (local, no Expo account)

```powershell
cd mobile
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build-apk.ps1 [-DefaultServer https://<name>.ngrok-free.app] [-Clean] [-UniversalOnly]
```

The script:
1. creates a release keystore on first run (`F:\models\keys\sightline-release.jks`, with a random password saved next to it in `sightline-release.properties`, outside the repo). **Back both files up.** Updates to an installed app must be signed with the same key.
2. runs `npx expo prebuild` to generate `android/` (gitignored, regenerated from `app.config.ts`).
3. runs `gradlew assembleRelease` (R8 and resource shrinking on, ABIs arm64-v8a, armeabi-v7a and x86_64).
4. writes two APKs, each with a `.sha256` file: `F:\sightline_release\sightline.apk` (universal: phones plus x86_64 emulators) and `sightline-arm64.apk` (arm64 phones only, which covers almost every phone since 2017, and is much smaller). `-UniversalOnly` skips the second one.

If the repo sits in a long folder, the script mirrors `mobile\` to a short folder (`F:\sightline_build\mobile`, set with `-BuildDir`) and builds there. Otherwise the native (CMake/ninja) steps fail on Windows' 260-character path limit. The mirror keeps its `android\` folder between runs, so later builds are incremental.

Run it detached if your terminal sessions are short, with output going to `F:\models\logs\mobile_build.log`, and check on it with `scripts\build-status.ps1`. The first build takes 10 to 20 minutes and later builds are faster.

**Cloud alternative (EAS):** `npm i -g eas-cli`, `eas login`, `eas init` (this sets the project ID; pass it as `EAS_PROJECT_ID`), then `eas build -p android --profile preview` with a profile that has `"android": {"buildType": "apk"}`. EAS manages the keystore for you.

## Exposing the backend (ngrok)

The phone reaches the backend through ngrok's free static domain. See the ngrok runbook in [`../backend/README.md`](../backend/README.md#ngrok-runbook). In short: `ngrok config add-authtoken <token>`, then `backend\scripts\start_ngrok.ps1`, and enter `https://<name>.ngrok-free.app` as the server in the app. The app sends `ngrok-skip-browser-warning`, so ngrok's interstitial page never appears.

The free plan allows 1 GB and 20,000 requests a month, so the app is frugal: snapshots load once per screen, the live view runs at 1 fps and stops after a session, alerts come over one WebSocket, and thumbnails and clips are cached and range-streamed.

## Installing it (judges)

1. On the Android phone, download `sightline-arm64.apk` from the link in the submission. If the phone is very old (32-bit) or you are using an emulator, use `sightline.apk` instead.
2. Open it and allow "Install unknown apps" for your browser or file manager when Android asks. Play Protect may warn that the app is from an unknown developer. Choose **More details, then Install anyway**.
3. Open Sightline and enter the server URL and the demo login from the submission notes.
4. Allow notifications to get hazard alerts.

Android 8.0 or later is supported (Android 13+ asks for notification permission).

## Notifications

- **Without Firebase (the default build):** alerts arrive over the WebSocket. While the app is open you get an in-app banner, and while it's in the background you get a local notification, for as long as Android keeps the app alive. Tapping either opens the event.
- **Remote push (app closed):** the backend already sends Expo push messages. For the phone to receive them, the build needs Firebase Cloud Messaging:
  1. Create a Firebase project and add an Android app with package `com.alienbtw.sightline`. Download `google-services.json` (keep it out of git) and build with `$env:SIGHTLINE_GOOGLE_SERVICES="C:\path\to\google-services.json"`.
  2. Create an Expo project (`eas init`) and build with `$env:EAS_PROJECT_ID="<id>"`. The app needs this ID to get an Expo push token.
  3. Upload the FCM V1 service-account key to the Expo project (`eas credentials`, Android, then Push Notifications).
  4. Rebuild. Settings then shows "Push: registered", and "Send test push" arrives even with the app closed.

## Project layout

```
mobile/
  app.config.ts          app name, icon, splash, permissions, plugins, runtime config
  plugins/               config plugins: cleartext only for loopback, release signing from env
  scripts/build-apk.ps1  local release build (build-status.ps1 checks a detached run)
  scripts/api-smoke.mts  API client smoke test against a live backend (Node)
  src/app/               screens (expo-router): login, (tabs)/cameras|alerts|settings, camera/[id], event/[id]
  src/components/        UI kit, MJPEG view, clip player with overlay, cards, sheets, in-app alert
  src/lib/               API client (auth + refresh), session, realtime, notifications, upload, overlay timing
  __tests__/             Jest unit tests
```