# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's private vulnerability reporting:
https://github.com/stfurkan/aidekin/security/advisories/new

Do not open a public issue for a security report. I aim to acknowledge within a few days.

## Threat model

aidekin runs entirely in the visitor's browser. There is no backend, no API keys, and no
server-side state, which removes whole classes of risk but also bounds what can be enforced:

- The system prompt, the configuration, and any `knowledge.bin` are delivered to the browser,
  so treat them as **public**. Never put secrets, credentials, or private data in them.
- Because the model runs locally, a determined visitor can inspect or alter the prompt and the
  model on their own machine. The system prompt is configuration, not a security boundary.
- The widget loads in a sandboxed cross-origin iframe and is driven only by the embedding page
  (and any `data-allowed-origins`) through origin-checked `postMessage`.
- Cross-origin isolation (COOP/COEP) is required for voice; see DEPLOY.md.

## Supported versions

This is a 0.x project. Security fixes target the latest commit on `main`.
