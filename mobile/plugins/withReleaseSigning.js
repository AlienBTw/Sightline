// Release signing without secrets in the repo: app/build.gradle reads the keystore path and passwords
// from Gradle properties (ORG_GRADLE_PROJECT_SIGHTLINE_* env vars set by scripts/build-apk.ps1 from
// F:\models\keys\sightline-release.properties). Without them, release falls back to the debug key.
const { withAppBuildGradle } = require('expo/config-plugins');

const RELEASE_CONFIG = `
        release {
            if (project.hasProperty('SIGHTLINE_STORE_FILE')) {
                storeFile file(project.property('SIGHTLINE_STORE_FILE'))
                storePassword project.property('SIGHTLINE_STORE_PASSWORD')
                keyAlias project.property('SIGHTLINE_KEY_ALIAS')
                keyPassword project.property('SIGHTLINE_KEY_PASSWORD')
            }
        }`;

module.exports = function withReleaseSigning(config) {
  return withAppBuildGradle(config, (cfg) => {
    let g = cfg.modResults.contents;
    if (!g.includes('SIGHTLINE_STORE_FILE')) {
      g = g.replace(/signingConfigs\s*\{/, (m) => `${m}${RELEASE_CONFIG}`);
      // in the release buildType, use the release key when it is configured
      g = g.replace(
        /(release\s*\{[^{}]*?)signingConfig\s+signingConfigs\.debug/,
        `$1signingConfig project.hasProperty('SIGHTLINE_STORE_FILE') ? signingConfigs.release : signingConfigs.debug`
      );
    }
    cfg.modResults.contents = g;
    return cfg;
  });
};
