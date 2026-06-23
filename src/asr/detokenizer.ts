// Browser-portable detokenizer for the Nemotron ASR Unigram SentencePiece vocab.
// The tokenizer.json has `decoder: null` and a normalizer that maps " " → "▁", so
// detok is the inverse: concatenate pieces, turn "▁" back into spaces, and drop
// the language-tag tokens (<en-US>, <tr-TR>, …) + <unk>/<blank> that the RNNT can emit.

export interface UnigramTokenizerJson {
  readonly model: { readonly vocab: ReadonlyArray<readonly [string, number]> }
}

const SPACE = '▁' // ▁ SentencePiece meta-space
const LANG_TAG = /^<[a-z]{2}-[A-Z]{2}>$/
const SPECIAL = /^<(unk|blank|pad|s|\/s)>$/

export class NemotronDetokenizer {
  private readonly idToPiece: readonly string[]
  private readonly drop: ReadonlySet<number>

  constructor(tokenizerJson: UnigramTokenizerJson) {
    const pieces = tokenizerJson.model.vocab.map((v) => v[0])
    const drop = new Set<number>()
    pieces.forEach((p, id) => {
      if (SPECIAL.test(p) || LANG_TAG.test(p)) drop.add(id)
    })
    this.idToPiece = pieces
    this.drop = drop
  }

  /** Full detokenization of a list of token ids → clean text. */
  decode(ids: readonly number[]): string {
    let out = ''
    for (const id of ids) {
      if (this.drop.has(id)) continue
      const p = this.idToPiece[id]
      if (p) out += p
    }
    return out.replaceAll(SPACE, ' ').replace(/\s+/g, ' ').trim()
  }

  /** Streaming fragment for one token (▁→space), or null if it should be hidden. */
  pieceFor(id: number): string | null {
    if (this.drop.has(id)) return null
    const p = this.idToPiece[id]
    return p ? p.replaceAll(SPACE, ' ') : null
  }
}
