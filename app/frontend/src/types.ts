export interface QueryResponse {
  question: string
  sql: string
  geojson: GeoJSON.FeatureCollection
  feature_count: number
}

export interface HistoryEntry {
  id: string
  question: string
  sql: string
  geojson: GeoJSON.FeatureCollection
  featureCount: number
  timestamp: number
}
