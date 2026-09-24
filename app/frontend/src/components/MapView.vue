<script setup lang="ts">
import { onBeforeUnmount, onMounted, shallowRef, watch } from 'vue'
// maplibre-gl v6 dropped its default export (named exports only) — MapLibreMap
// is the library's own alias for `Map`, to avoid shadowing JS's built-in Map.
import { GeoJSONSource, LngLatBounds, MapLibreMap, NavigationControl } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

const props = defineProps<{
  geojson: GeoJSON.FeatureCollection | null
}>()

const mapContainer = shallowRef<HTMLDivElement | null>(null)
let map: MapLibreMap | null = null
let resizeObserver: ResizeObserver | null = null

const SOURCE_ID = 'llaici-results'
const EMPTY_FC: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }

onMounted(() => {
  if (!mapContainer.value) return

  map = new MapLibreMap({
    container: mapContainer.value,
    // OpenFreeMap's "Liberty" style — free, no API key, full OSM detail
    // (countries, regions/states, cities/towns, roads, buildings...), unlike
    // MapLibre's own demotiles.maplibre.org style which only has coastlines and
    // country outlines. See https://openfreemap.org.
    style: 'https://tiles.openfreemap.org/styles/liberty',
    center: [10, 50], // roughly the middle of this project's EU bbox (see Makefile BBOX)
    zoom: 4,
  })

  // top-left, not top-right: the prompt panel overlay (App.vue) sits top-right.
  map.addControl(new NavigationControl(), 'top-left')

  // The container is sized by a flex layout (App.vue's `.layout`), so its final
  // pixel size isn't necessarily known the instant the map is constructed above
  // — MapLibre captures the container size once at init, and if that happened
  // to be 0 (or wrong) it won't repaint the vector tile layers on its own,
  // leaving only the style's solid-color background layer visible (a plain
  // blue screen, since that background represents ocean in this demo style).
  // A ResizeObserver keeps the map's internal size in sync with the container's
  // actual size whenever it changes, which also handles window resizes.
  resizeObserver = new ResizeObserver(() => map?.resize())
  resizeObserver.observe(mapContainer.value)

  map.on('load', () => {
    if (!map) return

    map.addSource(SOURCE_ID, { type: 'geojson', data: EMPTY_FC })

    // Three layers, one per geometry type that can come back from the DuckDB
    // views (SCHEMA.md: points for divisions/places, points/lines/polygons for
    // infrastructures/water) — MapLibre needs a layer per render type, it can't
    // style mixed-geometry sources with one layer. `['geometry-type']` returns
    // the exact GeoJSON type ("Polygon" vs "MultiPolygon" are distinct), and
    // Overture administrative boundaries in particular are very often
    // MultiPolygon (coastal/island regions), so each filter matches both the
    // single and Multi* variant.
    map.addLayer({
      id: `${SOURCE_ID}-polygons`,
      type: 'fill',
      source: SOURCE_ID,
      filter: ['in', ['geometry-type'], ['literal', ['Polygon', 'MultiPolygon']]],
      paint: { 'fill-color': '#2f6feb', 'fill-opacity': 0.35, 'fill-outline-color': '#2f6feb' },
    })

    map.addLayer({
      id: `${SOURCE_ID}-lines`,
      type: 'line',
      source: SOURCE_ID,
      filter: ['in', ['geometry-type'], ['literal', ['LineString', 'MultiLineString']]],
      paint: { 'line-color': '#e3342f', 'line-width': 3 },
    })

    // Soft halo behind each point, to make results stand out against the
    // basemap rather than looking like an ordinary map label/dot. Added before
    // the solid point layer below so it renders underneath it.
    map.addLayer({
      id: `${SOURCE_ID}-points-halo`,
      type: 'circle',
      source: SOURCE_ID,
      filter: ['in', ['geometry-type'], ['literal', ['Point', 'MultiPoint']]],
      paint: {
        'circle-radius': 16,
        'circle-color': '#e3342f',
        'circle-opacity': 0.25,
      },
    })

    map.addLayer({
      id: `${SOURCE_ID}-points`,
      type: 'circle',
      source: SOURCE_ID,
      filter: ['in', ['geometry-type'], ['literal', ['Point', 'MultiPoint']]],
      paint: {
        'circle-radius': 7,
        'circle-color': '#e3342f',
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff',
      },
    })

    updateData(props.geojson)
  })
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  resizeObserver = null
  map?.remove()
  map = null
})

function updateData(featureCollection: GeoJSON.FeatureCollection | null) {
  if (!map) return
  const source = map.getSource(SOURCE_ID) as GeoJSONSource | undefined
  if (!source) return

  const data = featureCollection ?? EMPTY_FC
  source.setData(data)

  if (data.features.length === 0) return

  // A single point result (the common "restaurants in Madrid" case with one
  // match, or the /query/mock fixture) gets a deliberate fly-to-city-level zoom
  // rather than fitBounds, which has no real extent to fit for a single point
  // and would otherwise depend on the maxZoom fallback to zoom in at all.
  if (data.features.length === 1 && data.features[0]?.geometry?.type === 'Point') {
    const [lng, lat] = data.features[0].geometry.coordinates as [number, number]
    map.flyTo({ center: [lng, lat], zoom: 12, duration: 900 })
    return
  }

  const bounds = new LngLatBounds()
  let hasCoords = false
  for (const feature of data.features) {
    if (!feature.geometry) continue
    extendBoundsForGeometry(bounds, feature.geometry)
    hasCoords = true
  }
  if (hasCoords) {
    map.fitBounds(bounds, { padding: 60, maxZoom: 15, duration: 600 })
  }
}

function extendBoundsForGeometry(bounds: LngLatBounds, geometry: GeoJSON.Geometry) {
  switch (geometry.type) {
    case 'Point':
      bounds.extend(geometry.coordinates as [number, number])
      break
    case 'MultiPoint':
    case 'LineString':
      for (const coord of geometry.coordinates) bounds.extend(coord as [number, number])
      break
    case 'MultiLineString':
    case 'Polygon':
      for (const ring of geometry.coordinates) {
        for (const coord of ring) bounds.extend(coord as [number, number])
      }
      break
    case 'MultiPolygon':
      for (const polygon of geometry.coordinates) {
        for (const ring of polygon) {
          for (const coord of ring) bounds.extend(coord as [number, number])
        }
      }
      break
    case 'GeometryCollection':
      for (const geom of geometry.geometries) extendBoundsForGeometry(bounds, geom)
      break
  }
}

watch(
  () => props.geojson,
  (featureCollection) => updateData(featureCollection),
)
</script>

<template>
  <div ref="mapContainer" class="map-container" />
</template>

<style scoped>
.map-container {
  width: 100%;
  height: 100%;
}
</style>
