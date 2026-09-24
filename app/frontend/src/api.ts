import type { QueryResponse } from './types'

// Same-origin in dev (proxied to the backend by vite.config.ts); point
// VITE_API_BASE at the real backend URL for a production build served
// separately from the API.
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function postQuery(path: string, question: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed with status ${res.status}`)
  }

  return res.json() as Promise<QueryResponse>
}

export async function askQuestion(question: string): Promise<QueryResponse> {
  return postQuery('/query', question)
}

// Temporary: hits the backend's canned "/query/mock" (always a single point in
// Bilbao) — lets the map's highlight/zoom behavior be built and tested without
// a running llama-server or fine-tuned model (see FINETUNING.md status). App.vue
// currently calls this instead of askQuestion; switch back once the real
// pipeline works end to end.
export async function askQuestionMock(question: string): Promise<QueryResponse> {
  return postQuery('/query/mock', question)
}
