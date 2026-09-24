<script setup lang="ts">
import { ref } from 'vue'
import type { HistoryEntry } from '../types'
import ConfirmModal from './ConfirmModal.vue'

const props = defineProps<{
  loading: boolean
  featureCount: number | null
  sql: string | null
  error: string | null
  history: HistoryEntry[]
  activeHistoryId: string | null
}>()

const emit = defineEmits<{
  submit: [question: string]
  'select-history': [id: string]
  'clear-history': []
}>()

const question = defineModel<string>('question', { default: '' })

const showClearConfirm = ref(false)

function onSubmit() {
  const trimmed = question.value.trim()
  if (!trimmed || props.loading) return
  emit('submit', trimmed)
}

function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function confirmClear() {
  showClearConfirm.value = false
  emit('clear-history')
}
</script>

<template>
  <aside class="panel">
    <header class="panel__header">
      <div class="panel__logo">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M12 21s-7-6.1-7-11.5A7 7 0 0 1 12 2.5a7 7 0 0 1 7 7C19 14.9 12 21 12 21Z"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linejoin="round"
          />
          <circle cx="12" cy="9.5" r="2.5" stroke="currentColor" stroke-width="1.8" />
        </svg>
      </div>
      <div>
        <h1 class="panel__title">LLaIci</h1>
        <p class="panel__subtitle">Ask, in plain language</p>
      </div>
    </header>

    <form class="panel__form" @submit.prevent="onSubmit">
      <div class="panel__field">
        <svg class="panel__field-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="1.8" />
          <path d="m20 20-3.5-3.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
        </svg>
        <textarea
          v-model="question"
          class="panel__textarea"
          placeholder="restaurants in Madrid, hotels near the station…"
          rows="2"
          :disabled="loading"
          @keydown.enter.exact.prevent="onSubmit"
        />
      </div>
      <button class="panel__submit" type="submit" :disabled="loading || !question.trim()">
        <span v-if="loading" class="panel__spinner" aria-hidden="true" />
        {{ loading ? 'Searching…' : 'Search' }}
      </button>
    </form>

    <Transition name="fade">
      <div v-if="error" class="panel__banner panel__banner--error">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8" />
          <path d="M12 8v5M12 16h.01" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
        </svg>
        <span>{{ error }}</span>
      </div>
    </Transition>

    <Transition name="fade">
      <div v-if="featureCount !== null && !error" class="panel__result">
        <span class="panel__count">{{ featureCount }}</span>
        <span class="panel__count-label">{{ featureCount === 1 ? 'result found' : 'results found' }}</span>
      </div>
    </Transition>

    <Transition name="fade">
      <details v-if="sql" class="panel__sql">
        <summary>
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="m9 18 6-6-6-6"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
          Generated SQL
        </summary>
        <pre>{{ sql }}</pre>
      </details>
    </Transition>

    <details v-if="history.length > 0" class="panel__history">
      <summary>
        <svg class="panel__history-chevron" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="m9 18 6-6-6-6"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        Recent searches
        <span class="panel__history-badge">{{ history.length }}</span>
        <button
          type="button"
          class="panel__history-clear"
          aria-label="Clear recent searches"
          title="Clear recent searches"
          @click.stop.prevent="showClearConfirm = true"
        >
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M4 7h16M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2m3 0-.87 12.14A2 2 0 0 1 15.14 21H8.86a2 2 0 0 1-1.99-1.86L6 7m4 4v6m4-6v6"
              stroke="currentColor"
              stroke-width="1.8"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        </button>
      </summary>
      <ul class="panel__history-list">
        <li v-for="entry in history" :key="entry.id">
          <button
            type="button"
            class="panel__history-item"
            :class="{ 'panel__history-item--active': entry.id === activeHistoryId }"
            @click="emit('select-history', entry.id)"
          >
            <span class="panel__history-question">{{ entry.question }}</span>
            <span class="panel__history-meta">
              <span class="panel__history-count">{{ entry.featureCount }}</span>
              <span class="panel__history-time">{{ formatTime(entry.timestamp) }}</span>
            </span>
          </button>
        </li>
      </ul>
    </details>

    <ConfirmModal
      :open="showClearConfirm"
      title="Clear recent searches?"
      message="This removes all saved search history from this browser. This can't be undone."
      confirm-label="Clear history"
      @confirm="confirmClear"
      @cancel="showClearConfirm = false"
    />
  </aside>
</template>

<style scoped>
.panel {
  --accent: #4f46e5;
  --accent-dark: #4338ca;
  --ink: #1f2430;
  --muted: #6b7280;
  --danger: #dc2626;
  --danger-bg: #fef2f2;

  display: flex;
  flex-direction: column;
  gap: 1.1rem;
  width: 340px;
  max-width: calc(100vw - 2.5rem);
  max-height: calc(100vh - 2.5rem);
  overflow-y: auto;
  padding: 1.4rem;
  box-sizing: border-box;
  background: rgba(255, 255, 255, 0.86);
  backdrop-filter: blur(14px) saturate(160%);
  -webkit-backdrop-filter: blur(14px) saturate(160%);
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.6);
  box-shadow:
    0 20px 40px -20px rgba(31, 36, 48, 0.35),
    0 2px 8px rgba(31, 36, 48, 0.08);
  color: var(--ink);
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    Roboto,
    sans-serif;
}

.panel__header {
  display: flex;
  align-items: center;
  gap: 0.7rem;
}

.panel__logo {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  flex-shrink: 0;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--accent), #818cf8);
  color: white;
}

.panel__logo svg {
  width: 20px;
  height: 20px;
}

.panel__title {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.panel__subtitle {
  margin: 0.1rem 0 0;
  color: var(--muted);
  font-size: 0.82rem;
}

.panel__form {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.panel__field {
  position: relative;
  display: flex;
}

.panel__field-icon {
  position: absolute;
  top: 0.65rem;
  left: 0.7rem;
  width: 17px;
  height: 17px;
  color: var(--muted);
  pointer-events: none;
}

.panel__textarea {
  flex: 1;
  resize: none;
  font-family: inherit;
  font-size: 0.92rem;
  line-height: 1.4;
  padding: 0.6rem 0.7rem 0.6rem 2.2rem;
  border: 1px solid #e2e4ea;
  border-radius: 6px;
  background: #f8f9fb;
  color: var(--ink);
  transition:
    border-color 0.15s ease,
    background 0.15s ease;
}

.panel__textarea::placeholder {
  color: #9ca0ab;
}

.panel__textarea:focus {
  outline: none;
  border-color: var(--accent);
  background: #ffffff;
}

.panel__textarea:disabled {
  opacity: 0.6;
}

.panel__submit {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.65rem 1rem;
  border: none;
  border-radius: 6px;
  background: var(--accent);
  color: white;
  font-weight: 600;
  font-size: 0.92rem;
  cursor: pointer;
  transition:
    background 0.15s ease,
    transform 0.1s ease;
}

.panel__submit:hover:not(:disabled) {
  background: var(--accent-dark);
}

.panel__submit:active:not(:disabled) {
  transform: scale(0.98);
}

.panel__submit:disabled {
  background: #c7cbf5;
  cursor: not-allowed;
}

.panel__spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.panel__banner {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  padding: 0.65rem 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
  line-height: 1.4;
}

.panel__banner svg {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  margin-top: 0.05rem;
}

.panel__banner--error {
  background: var(--danger-bg);
  color: var(--danger);
}

.panel__result {
  display: flex;
  align-items: baseline;
  gap: 0.45rem;
  padding: 0.15rem 0.1rem;
}

.panel__count {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--accent);
  letter-spacing: -0.02em;
}

.panel__count-label {
  color: var(--muted);
  font-size: 0.88rem;
}

.panel__sql {
  border-top: 1px solid #eceef2;
  padding-top: 0.7rem;
}

.panel__sql summary {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  cursor: pointer;
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 500;
  list-style: none;
  user-select: none;
}

.panel__sql summary::-webkit-details-marker {
  display: none;
}

.panel__sql summary svg {
  width: 14px;
  height: 14px;
  transition: transform 0.15s ease;
}

.panel__sql[open] summary svg {
  transform: rotate(90deg);
}

.panel__sql pre {
  margin: 0.55rem 0 0;
  white-space: pre-wrap;
  word-break: break-word;
  background: #0f1117;
  color: #d5d9e2;
  padding: 0.7rem 0.8rem;
  border-radius: 6px;
  font-size: 0.76rem;
  line-height: 1.5;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

.panel__history {
  border-top: 1px solid #eceef2;
  padding-top: 0.7rem;
}

.panel__history summary {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  cursor: pointer;
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 500;
  list-style: none;
  user-select: none;
}

.panel__history summary::-webkit-details-marker {
  display: none;
}

.panel__history summary .panel__history-chevron {
  width: 14px;
  height: 14px;
  transition: transform 0.15s ease;
}

.panel__history[open] summary .panel__history-chevron {
  transform: rotate(90deg);
}

.panel__history-badge {
  margin-left: auto;
  padding: 0.05rem 0.45rem;
  border-radius: 4px;
  background: #eef0f4;
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 600;
}

.panel__history-clear {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.2rem;
  border: none;
  border-radius: 4px;
  background: none;
  color: var(--muted);
  cursor: pointer;
  transition:
    color 0.12s ease,
    background 0.12s ease;
}

.panel__history-clear svg {
  width: 15px;
  height: 15px;
}

.panel__history-clear:hover {
  color: var(--danger);
  background: var(--danger-bg);
}

.panel__history-list {
  list-style: none;
  margin: 0.5rem 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.panel__history-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  width: 100%;
  padding: 0.5rem 0.6rem;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  color: var(--ink);
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition: background 0.12s ease;
}

.panel__history-item:hover {
  background: #f2f2fb;
}

.panel__history-item--active {
  background: #eef0fd;
  border-color: #d8dcf9;
}

.panel__history-question {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.85rem;
}

.panel__history-meta {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-shrink: 0;
  font-size: 0.72rem;
  color: var(--muted);
}

.panel__history-count {
  padding: 0.05rem 0.4rem;
  border-radius: 4px;
  background: #e5e7ff;
  color: var(--accent-dark);
  font-weight: 600;
}

.fade-enter-active,
.fade-leave-active {
  transition:
    opacity 0.18s ease,
    transform 0.18s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
