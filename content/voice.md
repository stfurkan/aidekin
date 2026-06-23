# Voice and text

aidekin supports both text and voice. Choose the mode in the configurator at https://aidekin.com/configure, or set data-mode on the script tag: "both" for text with a voice toggle, or "voice" for a voice-first experience. Text works on any WebGPU browser with no special setup.

Voice uses the same language model as text, plus on-device speech recognition and speech synthesis. The same brain answers in both modes, so the conversation context carries over.

## What downloads, and when

Text chat downloads about 290 MB once: the language model. It is then cached and reused on later visits.

Turning on voice adds about 1.6 GB of speech models, but only the first time a visitor taps the microphone, never before. Those are cached too. After the first load, aidekin works offline.

## Cross-origin isolation

Voice's fastest path uses threaded WebAssembly, which runs best when the embedding page is cross-origin isolated (the COOP and COEP headers). On a page that is not isolated, voice runs single-threaded if it can keep up, and otherwise stays in text mode. Text is never affected.
