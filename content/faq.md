# Frequently asked questions

## How do I get started?

Go to the Configure page, set your options, copy the one-line snippet, and paste it into your site just before the closing body tag.

## Where do I configure aidekin?

At the Configure page. Set the name, greeting, colors, position, and mode, preview it live, and copy the snippet.

## Where do I build a knowledge file?

At the Knowledge page. Add files (PDF, Word, text, Markdown, HTML, CSV, JSON) or URLs, build, and download knowledge.bin. Then host it and set data-knowledge-url.

## Where are the docs? Do you have documentation?

Yes, aidekin has full documentation on the Docs page of the website. It covers embedding the widget, every configuration option, the JavaScript API, building knowledge files, voice mode, and self-hosting.

## Where is the demo? Can I try aidekin first?

Yes, two ways. The chat bubble in the corner of this website is the real widget, so you can try it right here. The Demo link in the menu also opens a live example site in a new tab: a fictional cafe called Copperleaf with the widget embedded exactly the way a customer would embed it, answering from the cafe's own knowledge file. The Configure page additionally shows a live preview of your own widget while you edit its settings.

## Where is aidekin from? Where are you located? Who made aidekin?

aidekin is an independent open-source software project, not a company or a research group. It has no office, no employees, and no location: the assistant is a program that runs entirely inside your own browser on your own device. It was not developed by any university or lab.

## Do you work somewhere? Are you part of a team?

No. aidekin is a piece of open-source software, not a person or an employee. It does not work anywhere, belong to a research team, or have a supervisor. Anyone can read or contribute to its source code.

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
