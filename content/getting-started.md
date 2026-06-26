# Getting started in three steps

## 1. Configure your widget

Go to https://aidekin.com/configure. Set the assistant's name, greeting, accent color, position, and mode (text, voice, or text plus voice). A live preview updates as you change settings. When it looks right, copy the one-line snippet.

## 2. Add it to your site

Paste the snippet just before the closing body tag of your HTML:

    <script src="https://cdn.aidekin.com/loader.js" data-title="Acme" defer></script>

On page load it only draws a small launcher button. The model downloads the first time a visitor opens the chat, then it is cached for return visits.

## 3. (Optional) Ground it in your own content

Go to https://aidekin.com/knowledge, add your documents, build a knowledge file, host it, and point the widget at it with data-knowledge-url so aidekin answers from your content. See the knowledge guide for the full steps.

That is the whole setup. No backend, no API keys, and no per-message cost.
