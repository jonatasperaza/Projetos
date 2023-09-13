<script setup>
import { ref } from 'vue'
const n1 = 1
const n2 = 2
const n3 = 3
const n4 = 4
const n5 = 5
const n6 = 6
const n7 = 7
const n8 = 8
const n9 = 9
let t = 0

</script>

<template>
  <div class="calculadora">
    <div class="visor">
      <p>numeros</p>
    </div>
    <div class="botoes">
      <div class="botoes_numeros">
        <ol>
          <button @click="n1">1</button>
          <button @click="n2">2</button>
          <button @click="n3">3</button>
          <button @click="n4">4</button>
          <button @click="n5">5</button>
          <button @click="n6">6</button>
          <button @click="n7">7</button>
          <button @click="n8">8</button>
          <button @click="n9">9</button>
        </ol>
      </div>
      <div class="botoes_operacoes">
        <ol>
          <button @click="mais">+</button>
          <button @click="menos">-</button>
          <button @click="vezes">*</button>
          <button @click="dividi">/</button>
          
        </ol>
      </div>
    </div>
  </div>
</template>

<style scoped>
.calculadora {
  border: 1px solid black;
}
</style>
