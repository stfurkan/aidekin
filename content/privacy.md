# Privacy and security

Everything in aidekin runs on the visitor's device. There is no backend, no API key, and no per-message cost, so there is nothing on your side for a visitor to steal or run up.

What aidekin enforces:

- The system prompt is configuration, not a secret. It is delivered to the browser. Set the persona, but never put credentials or private instructions in it.
- The knowledge file is public. Every visitor downloads it, so keep secrets and personal data out of it.
- The widget runs in a sandboxed iframe and only requests the microphone when voice is used.
- Cross-origin messages are checked, so only the embedding page (plus any origins you list in data-allowed-origins) can drive the widget.

The honest limit: because everything runs locally, a determined visitor can inspect or alter the prompt, config, and model on their own machine. There is nothing to lose at your expense, but do not treat the system prompt as a security boundary.
