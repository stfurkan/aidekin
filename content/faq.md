# Frequently asked questions

## How do I get started?

Go to https://aidekin.com/configure, set your options, copy the one-line snippet, and paste it into your site just before the closing body tag.

## Where do I configure aidekin?

At https://aidekin.com/configure. Set the name, greeting, colors, position, and mode, preview it live, and copy the snippet.

## Where do I build a knowledge file?

At https://aidekin.com/knowledge. Add files (PDF, Word, text, Markdown, HTML, CSV, JSON) or URLs, build, and download knowledge.bin. Then host it and set data-knowledge-url.

## Where are the docs?

At https://aidekin.com/docs.

## Is aidekin really free?

Yes. aidekin is open source under the MIT license. The model streams once from a public CDN and is cached in the visitor's browser. There are no per-message charges and no servers to run.

## What does a visitor download?

About 290 MB for text chat, once, then cached. Turning on voice adds about 1.6 GB the first time it is used. After that it works offline.

## Which browsers are supported?

aidekin needs WebGPU. That means recent desktop Chrome and Edge, Safari 26 and later, and Firefox 145 and later. On an unsupported browser, the widget shows a short, friendly notice instead of failing.

## Where does my data go?

Nowhere. Inference and retrieval happen entirely in the visitor's browser. Nothing is sent to a server.

## Can different sites share the cached model?

Browsers partition storage by site, so a visitor downloads the model once per site that embeds aidekin, then loads it from cache on return visits. This is a browser privacy protection, not a bug.

## Does the widget have a dark mode?

Yes. It follows the visitor's system theme by default, and visitors can switch light or dark themselves using the sun and moon button next to the settings menu. Owners can set a default with data-theme (light, dark, or auto).

## Can I self-host everything?

Yes. Build the project, deploy the output to any static host, and point the loader at your own copy with data-widget-origin. The model weights can stream from the public CDN or from your own mirror.
