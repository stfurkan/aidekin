# Knowledge files and RAG

To make aidekin answer from your own content, build a knowledge file and point the widget at it with data-knowledge-url. Retrieval runs in the browser, and at query time only the visitor's question is embedded. If you do not set a knowledge URL, no retrieval code or embedder ever loads.

## Where to build a knowledge file

The easiest way is the in-browser builder at https://aidekin.com/knowledge. Open it, add your content (drag in files, paste text, or add page URLs), then click Build and download to get a small knowledge.bin file. You can upload PDF, Word, plain text, Markdown, HTML, CSV, and JSON files.

## Advanced: build from the command line (optional)

Developers can produce the same file from the command line. Pass your input folder and output path as arguments; the output is identical to the browser builder:

    npm run build-knowledge -- --in ./docs --out ./public/knowledge.bin

## How to host the knowledge file

Host the small file anywhere that allows cross-origin reads:

- A public GitHub repo, served free through a CDN like jsDelivr.
- Cloudflare R2, GitHub Pages, Netlify, or Vercel (all have free tiers).
- Your own server or bucket, with the response header Access-Control-Allow-Origin set to a star.

These are independent services; aidekin is not affiliated with any of them.

## How to connect it

Add the file's URL to your snippet as data-knowledge-url. The configurator at https://aidekin.com/configure adds it for you. Tune how many chunks are retrieved with data-rag-top-k (the default is 3).

The knowledge file is downloaded by every visitor, so treat it as public and never put secrets in it.

## Does a custom system prompt affect retrieval?

No. A custom system prompt, set in the configurator, changes the assistant's persona and tone only. It does not turn off retrieval. When a knowledge URL is set, answers are still grounded in your knowledge file regardless of the system prompt.
