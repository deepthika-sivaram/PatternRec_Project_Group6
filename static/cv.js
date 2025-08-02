// static/cv.js

// 1. Class mappings
const PRODUCE_CLASSES = ['Apple', 'Grapes', 'Peach', 'Raspberry'];
const VARIATION_CLASSES = {
  'Apple': ['Whole', 'Sliced-Cored', 'In-Context'],
  'Grapes': ['In a Bag', 'Loose Grapes', 'On the Vine'],
  'Peach': ['Halved or Pitted', 'Sliced', 'Whole'],
  'Raspberry': ['In a Container', 'Slightly Crushed', 'Small Group']
};

// 2. ONNX Runtime sessions
let produceSession;
let variationSessions = {};

export async function initModels() {
  console.log('🔄 Loading ONNX models...');
  produceSession = await ort.InferenceSession.create('/static/models/produce.onnx');
  for (let fruit of PRODUCE_CLASSES) {
    variationSessions[fruit] =
      await ort.InferenceSession.create(`/static/models/variation_${fruit}.onnx`);
  }
  console.log('✅ Models loaded.');
}

// 3. Convert image to tensor
function imgToTensor(img) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, 224, 224);
  const { data } = ctx.getImageData(0, 0, 224, 224);
  const arr = new Float32Array(1 * 3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    for (let ch = 0; ch < 3; ch++) {
      arr[ch * 224 * 224 + i] = data[i * 4 + ch] / 255.0;
    }
  }
  return new ort.Tensor('float32', arr, [1, 3, 224, 224]);
}

// 4. Classify produce and variation
export async function classifyAll(imgEl) {
  try {
    const tensor = imgToTensor(imgEl);

    // Produce
    const { output: pOut } = await produceSession.run({ input: tensor });
    const pIdx = pOut.data.indexOf(Math.max(...pOut.data));
    const fruit = PRODUCE_CLASSES[pIdx];
    document.getElementById('produce-label').innerText = fruit;

    // Variation
    const { output: vOut } = await variationSessions[fruit].run({ input: tensor });
    const vIdx = vOut.data.indexOf(Math.max(...vOut.data));
    const variation = VARIATION_CLASSES[fruit][vIdx];
    document.getElementById('variation-label').innerText = variation;
  } catch (err) {
    console.error('❌ Error during classification:', err);
  }
}

// 5. Bind event listeners
export function wireUp() {
  const input = document.getElementById('img-upload');
  const img = document.getElementById('preview');
  input.addEventListener('change', e => {
    img.src = URL.createObjectURL(e.target.files[0]);
    img.onload = () => classifyAll(img);
  });
}

// 6. Auto-init on DOM ready
window.addEventListener('DOMContentLoaded', async () => {
  await initModels();
  wireUp();
});
