// Live view: the backend streams ONE long MJPEG response (1 fps, 480 px, circles drawn server-side).
// Android's <Image> cannot decode MJPEG, so a locked-down WebView shows it in an <img>.
// The URL is the backend's signed, short-lived ?st= link (no auth header needed).
// The page is given the server's origin as its base URL: the backend sends
// Cross-Origin-Resource-Policy: same-origin, so a cross-origin page could not show the stream.
import { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import { WebView } from 'react-native-webview';
import { C } from '../lib/theme';
import { originOf } from '../lib/url';

function escapeAttr(s: string) {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function mjpegHtml(url: string) {
  return `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>html,body{margin:0;height:100%;background:#000;overflow:hidden}
body{display:flex;align-items:center;justify-content:center}
img{width:100%;height:100%;object-fit:contain}</style></head>
<body><img id="v" alt="" src="${escapeAttr(url)}"
 onload="window.ReactNativeWebView&&window.ReactNativeWebView.postMessage('frame')"
 onerror="window.ReactNativeWebView&&window.ReactNativeWebView.postMessage('error')"></body></html>`;
}

export function MjpegView({
  url,
  onFrame,
  onError,
}: {
  url: string;
  onFrame?: () => void;
  onError?: () => void;
}) {
  const html = useMemo(() => mjpegHtml(url), [url]);
  const origin = useMemo(() => originOf(url), [url]);
  return (
    <View style={styles.box}>
      <WebView
        source={{ html, baseUrl: origin + '/' }}
        originWhitelist={[origin]}
        // Non-browser UA: ngrok's free tier shows an HTML warning page to browsers.
        userAgent="SightlineApp/1.0 (Android; live-view)"
        // Only the inline page itself may load; no navigation anywhere else.
        onShouldStartLoadWithRequest={(r) => r.url === origin + '/' || r.url.startsWith('about:')}
        setSupportMultipleWindows={false}
        javaScriptEnabled
        domStorageEnabled={false}
        allowFileAccess={false}
        allowsInlineMediaPlayback
        mixedContentMode="never"
        cacheEnabled={false}
        incognito
        scrollEnabled={false}
        overScrollMode="never"
        style={styles.web}
        containerStyle={styles.web}
        onMessage={(e) => (e.nativeEvent.data === 'error' ? onError?.() : onFrame?.())}
        onError={() => onError?.()}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  box: { flex: 1, backgroundColor: '#000' },
  web: { flex: 1, backgroundColor: '#000' },
});

export const MJPEG_BG = C.bg;
