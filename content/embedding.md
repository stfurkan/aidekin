# Embedding aidekin

The easiest way to get your snippet is the configurator at the Configure page: set your options, watch the live preview, and copy the one-line tag. Paste it into your site just before the closing body tag:

    <script src="https://cdn.aidekin.com/loader.js" data-title="Acme" defer></script>

On page load the loader (about 2 KB) only draws a floating launcher button. The widget and the model load on the first open, so there is zero impact on your page load. The model downloads once and is cached for repeat visits.

## Configuration options

Configure aidekin with data attributes on the script tag, or with a window.AidekinConfig object set before the loader runs.

- data-mode: text, voice, or both. Default is text. "both" is text with a voice toggle.
- data-title: the assistant's name and panel header. Default is Assistant.
- data-greeting: the first message shown before the visitor types.
- data-system-prompt: the persona. It is delivered to the browser, so it is not a secret.
- data-accent: a CSS color for the launcher, send button, and assistant bubble.
- data-position: bottom-right or bottom-left.
- data-launcher-label: the text on the floating button.
- data-knowledge-url: the URL of a knowledge file for grounded answers (RAG).
- data-rag-top-k: how many knowledge chunks to retrieve. Default is 3.
- data-reasoning: true to make the model reason on every reply (slower but more accurate).
- data-persist: true to remember the conversation across reloads. Default is true.
- data-theme: light, dark, or auto. Default auto, which follows the visitor's system theme.

## JavaScript API

Once loaded, window.Aidekin exposes:
- open(), close(), and toggle() to control the panel from your own buttons.
- setTheme('light' or 'dark') to match your site's own light/dark toggle.
- on(event, callback) for the open, close, ready, and message events. The message event fires for each user and assistant turn with its role and text.

## Theme

The widget runs in a sandboxed iframe, so it cannot read your page's CSS. By default (data-theme auto) it follows the visitor's operating-system color scheme. To keep it in sync with your own theme switch, call window.Aidekin.setTheme('dark') or window.Aidekin.setTheme('light').

Visitors can also switch light or dark themselves using the sun and moon button next to the widget's settings menu, and their choice is saved for next time.
