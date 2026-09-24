<script setup lang="ts">
withDefaults(
  defineProps<{
    open: boolean
    title: string
    message: string
    confirmLabel?: string
    cancelLabel?: string
  }>(),
  {
    confirmLabel: 'Confirm',
    cancelLabel: 'Cancel',
  },
)

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()
</script>

<template>
  <!-- Teleported to body: this modal must overlay the whole viewport regardless
  of the prompt panel's own position/overflow (it's an absolutely-positioned,
  scrollable card — a modal nested inside it would be clipped or mispositioned). -->
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="open" class="modal-backdrop" @click.self="emit('cancel')">
        <div class="modal" role="alertdialog" aria-modal="true">
          <h2 class="modal__title">{{ title }}</h2>
          <p class="modal__message">{{ message }}</p>
          <div class="modal__actions">
            <button type="button" class="modal__button modal__button--ghost" @click="emit('cancel')">
              {{ cancelLabel }}
            </button>
            <button type="button" class="modal__button modal__button--danger" @click="emit('confirm')">
              {{ confirmLabel }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-backdrop {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 17, 23, 0.4);
}

.modal {
  width: 320px;
  max-width: calc(100vw - 2.5rem);
  padding: 1.3rem;
  background: #ffffff;
  border-radius: 10px;
  box-shadow: 0 24px 48px -16px rgba(15, 17, 23, 0.4);
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    Roboto,
    sans-serif;
}

.modal__title {
  margin: 0 0 0.4rem;
  font-size: 1.05rem;
  font-weight: 700;
  color: #1f2430;
}

.modal__message {
  margin: 0 0 1.1rem;
  font-size: 0.88rem;
  line-height: 1.45;
  color: #6b7280;
}

.modal__actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
}

.modal__button {
  padding: 0.5rem 0.9rem;
  border: none;
  border-radius: 6px;
  font: inherit;
  font-weight: 600;
  font-size: 0.85rem;
  cursor: pointer;
}

.modal__button--ghost {
  background: #f2f3f5;
  color: #1f2430;
}

.modal__button--ghost:hover {
  background: #e7e8eb;
}

.modal__button--danger {
  background: #dc2626;
  color: white;
}

.modal__button--danger:hover {
  background: #b91c1c;
}

.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.15s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}
</style>
