const n = Number(prompt("Me diga o primeiro numeros: "));
const s = prompt("Me diga o sinal (ex:+ - / * ou x)").toLowerCase();
const n2 = Number(prompt("Me diga o segundo numero"));
let n3 = Number(0)
if (isNaN(s) && isNaN(n) == false && isNaN(n2) == false) {
    if (s == "+"){
        n3 = n + n2
        alert(`${n3}`)
    }
} else {
  alert(`Digite APENAS os sinais `);
}
