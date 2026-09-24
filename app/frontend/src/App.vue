<script setup lang="ts">
import { ref, watch } from 'vue'
import MapView from './components/MapView.vue'
import PromptPanel from './components/PromptPanel.vue'
import { askQuestionMock } from './api'
import { loadHistory, saveHistory } from './history'
import type { HistoryEntry, QueryResponse } from './types'

const question = ref('')
const geojson = ref<GeoJSON.FeatureCollection | null>(null)
const sql = ref<string | null>(null)
const featureCount = ref<number | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const history = ref<HistoryEntry[]>(loadHistory())
const activeHistoryId = ref<string | null>(null)

watch(history, (entries) => saveHistory(entries), { deep: true })

async function handleSubmit(submittedQuestion: string) {
  loading.value = true
  error.value = null
  try {
    // TEMPORARY: using the mock endpoint until a real llama-server + fine-tuned
    // model exist (see FINETUNING.md status) — swap for askQuestion once they do.
    const result: QueryResponse = await askQuestionMock(submittedQuestion)
    geojson.value = result.geojson
    sql.value = result.sql
    featureCount.value = result.feature_count

    const entry: HistoryEntry = {
      id: crypto.randomUUID(),
      question: result.question,
      sql: result.sql,
      geojson: result.geojson,
      featureCount: result.feature_count,
      timestamp: Date.now(),
    }
    activeHistoryId.value = entry.id
    history.value = [entry, ...history.value]
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Something went wrong'
    featureCount.value = null
    activeHistoryId.value = null
  } finally {
    loading.value = false
  }
}

function handleSelectHistory(id: string) {
  const entry = history.value.find((h) => h.id === id)
  if (!entry) return

  question.value = entry.question
  geojson.value = entry.geojson
  sql.value = entry.sql
  featureCount.value = entry.featureCount
  error.value = null
  activeHistoryId.value = entry.id
}

function handleClearHistory() {
  // Only wipes the saved list — leaves whatever's currently on the map/panel
  // untouched, so clearing history doesn't feel like it also cleared the
  // current result.
  history.value = []
  activeHistoryId.value = null
}
</script>

<template>
  <div class="layout">
    <MapView class="layout__map" :geojson="geojson" />
    <PromptPanel
      v-model:question="question"
      class="layout__panel"
      :loading="loading"
      :feature-count="featureCount"
      :sql="sql"
      :error="error"
      :history="history"
      :active-history-id="activeHistoryId"
      @submit="handleSubmit"
      @select-history="handleSelectHistory"
      @clear-history="handleClearHistory"
    />
  </div>
</template>

<style scoped>
.layout {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}

.layout__map {
  position: absolute;
  inset: 0;
}

.layout__panel {
  position: absolute;
  top: 1.25rem;
  right: 1.25rem;
  z-index: 1;
}
</style>
