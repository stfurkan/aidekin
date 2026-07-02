// Standalone tokenizer for the LLM, replacing @huggingface/transformers' AutoTokenizer. Wraps the
// canonical HF tokenizer (@huggingface/tokenizers, the same byte-level BPE used inside transformers.js)
// and the canonical chat-template engine (@huggingface/jinja). Verified byte-exact against
// transformers.js for our model (scripts/verify-tokenizer.ts). The engine itself is tokenizer-free and
// works on token ids; this module is the product's encode/decode/chat-template layer.
import { Tokenizer } from '@huggingface/tokenizers'
import { Template } from '@huggingface/jinja'
import { getModelAsset } from './modelStore'

export interface ChatMessage {
  role: string
  content: string
}

/** Incremental decoder for streaming generation: feed token ids as they arrive, get the new visible
 *  text each step. Re-decodes the running sequence and emits the stable delta, holding back a trailing
 *  incomplete multi-byte character (which the byte-level decoder renders as U+FFFD) until it completes. */
export interface DecoderStream {
  push(tokenId: number): string
  flush(): string
}

export class LlmTokenizer {
  private readonly tok: Tokenizer
  private readonly template: Template | null
  /** End-of-sequence token id (e.g. <|im_end|> for Qwen3). */
  readonly eosTokenId: number

  constructor(tokenizerJson: unknown, tokenizerConfig: Record<string, unknown>) {
    this.tok = new Tokenizer(tokenizerJson as never, tokenizerConfig as never)
    const tmpl = tokenizerConfig['chat_template']
    this.template = typeof tmpl === 'string' ? new Template(tmpl) : null
    const eos = tokenizerConfig['eos_token']
    const eosStr = typeof eos === 'string' ? eos : ((eos as { content?: string } | undefined)?.content ?? '<|im_end|>')
    this.eosTokenId = this.tok.token_to_id(eosStr) ?? 151645
  }

  /** Encode text to token ids. `addSpecialTokens` defaults to false (the chat template already
   *  inserts the control tokens, so prompt/delta encoding must not add more). */
  encode(text: string, addSpecialTokens = false): number[] {
    return Array.from(this.tok.encode(text, { add_special_tokens: addSpecialTokens }).ids, Number)
  }

  /** Decode token ids to text. `skipSpecialTokens` defaults to true (never surface control tokens). */
  decode(ids: number[], skipSpecialTokens = true): string {
    return this.tok.decode(ids, { skip_special_tokens: skipSpecialTokens })
  }

  /** Render a chat message list to a prompt string via the model's own Jinja chat template
   *  (matches transformers.js apply_chat_template exactly). */
  applyChatTemplate(messages: ChatMessage[], opts: { addGenerationPrompt?: boolean; enableThinking?: boolean } = {}): string {
    if (!this.template) throw new Error('LlmTokenizer: model has no chat_template')
    return this.template.render({
      messages,
      add_generation_prompt: opts.addGenerationPrompt ?? true,
      enable_thinking: opts.enableThinking ?? false,
    }) as string
  }

  createDecoderStream(skipSpecialTokens = true): DecoderStream {
    const ids: number[] = []
    let emitted = 0
    const decodeAll = (): string => this.tok.decode(ids, { skip_special_tokens: skipSpecialTokens })
    return {
      push: (tokenId: number): string => {
        ids.push(tokenId)
        const text = decodeAll()
        let safe = text.length
        while (safe > emitted && text.charCodeAt(safe - 1) === 0xfffd) safe-- // hold back incomplete trailing char
        const out = text.slice(emitted, safe)
        emitted = safe
        return out
      },
      flush: (): string => {
        const text = decodeAll()
        const out = text.slice(emitted)
        emitted = text.length
        return out
      },
    }
  }

  /** Load tokenizer.json + tokenizer_config.json (from the HF Hub by model id, or explicit URLs). The
   *  caller may pass a `fetchJson` that overrides fetching; the default routes through the OPFS model
   *  cache (keyed by model id + filename), so a fully cached model still boots when the Hub is
   *  unreachable. The ~7MB tokenizer.json otherwise re-downloads on EVERY worker init. */
  static async load(
    source: { modelId: string } | { tokenizerJsonUrl: string; tokenizerConfigUrl: string },
    fetchJson?: (url: string) => Promise<unknown>,
  ): Promise<LlmTokenizer> {
    let jsonUrl: string
    let cfgUrl: string
    if ('modelId' in source) {
      const base = `https://huggingface.co/${source.modelId}/resolve/main`
      jsonUrl = `${base}/tokenizer.json`
      cfgUrl = `${base}/tokenizer_config.json`
    } else {
      jsonUrl = source.tokenizerJsonUrl
      cfgUrl = source.tokenizerConfigUrl
    }
    const get =
      fetchJson ??
      (async (url: string): Promise<unknown> => {
        const key = 'modelId' in source ? `llm-tokenizer/${source.modelId}/${url.split('/').pop()}` : `llm-tokenizer/${url}`
        return JSON.parse(new TextDecoder().decode(await getModelAsset(key, url)))
      })
    const [json, cfg] = await Promise.all([get(jsonUrl), get(cfgUrl)])
    return new LlmTokenizer(json, cfg as Record<string, unknown>)
  }
}
