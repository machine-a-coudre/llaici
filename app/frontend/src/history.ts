import type { HistoryEntry } from './types'

const STORAGE_KEY = 'llaici-history'
const MAX_ENTRIES = 50

// localStorage is per-browser, best-effort only (private browsing, quota, etc.
// can all make it throw or silently no-op) — never let it break the app.
export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveHistory(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)))
  } catch {
    // Ignore (quota exceeded, private mode, etc.) — history just won't persist.
  }
}
