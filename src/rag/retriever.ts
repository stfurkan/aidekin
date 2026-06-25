// Runtime retriever: fetches the owner's precomputed knowledge.bin, then for each query
// embeds it (WASM) and returns the top-k chunks. Implements the engine's Retriever
// interface. Dynamically imported only when a knowledgeUrl is configured.

import type { Retriever, RetrievedChunk } from '@/engine/conversationEngine'
import { embedQuery } from './embedder'
import { VectorStore } from './store'

export interface RetrieverInfo {
  retriever: Retriever
  /** Number of chunks in the loaded index (for UI / debugging). */
  count: number
}

export async function createRetriever(knowledgeUrl: string): Promise<RetrieverInfo> {
  const res = await fetch(knowledgeUrl)
  if (!res.ok) throw new Error(`Failed to fetch knowledge file (${res.status})`)
  const store = VectorStore.fromBytes(await res.arrayBuffer())

  const retriever: Retriever = {
    async retrieve(query: string, k: number): Promise<RetrievedChunk[]> {
      const qvec = await embedQuery(query)
      return store.search(qvec, k).map((h) => ({ text: h.text, score: h.score, source: h.source }))
    },
  }
  return { retriever, count: store.count }
}
