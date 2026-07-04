// Flat ESLint config: typescript-eslint's TYPE-CHECKED recommended set + React hooks rules.
// The type-checked rules are the valuable ones here (unhandled promises are the #1 real bug
// class in a worker-heavy codebase); the codebase's `void promise` idiom is the sanctioned
// way to mark intentional fire-and-forget.
import eslint from '@eslint/js'
import reactHooks from 'eslint-plugin-react-hooks'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      '.cache/**',
      'public/**',
      // generated + repo-external artifacts
      '**/*.generated.ts',
      // plain-JS one-offs that tsc doesn't cover (worklet runs inside AudioWorklet scope)
      'src/audio/pcmWorklet.js',
      'scripts/*.mjs',
    ],
  },
  eslint.configs.recommended,
  tseslint.configs.recommendedTypeChecked,
  reactHooks.configs.flat['recommended-latest'],
  { files: ['eslint.config.js'], extends: [tseslint.configs.disableTypeChecked] },
  {
    languageOptions: {
      parserOptions: {
        projectService: { allowDefaultProject: ['eslint.config.js'] },
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // `void somePromise` is this codebase's explicit fire-and-forget marker.
      '@typescript-eslint/no-floating-promises': ['error', { ignoreVoid: true }],
      // Worker protocols narrow unknown payloads; underscore-prefixed = intentionally unused.
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
)
