// https://docs.expo.dev/guides/using-eslint/
const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

module.exports = defineConfig([
  expoConfig,
  {
    ignores: ['dist/*', 'android/*', 'ios/*', 'scripts/*'],
  },
  {
    // TypeScript already checks namespace members; this rule misreads expo-notifications' re-exports.
    rules: { 'import/namespace': 'off' },
  },
]);
