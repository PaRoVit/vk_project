let tf;

const TF_CDN = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';

function loadTf() {
  return new Promise((resolve, reject) => {
    if (window.tf) { resolve(window.tf); return; }
    const script = document.createElement('script');
    script.src = TF_CDN;
    script.onload = () => resolve(window.tf);
    script.onerror = () => reject(new Error('Failed to load TensorFlow.js from CDN'));
    document.head.appendChild(script);
  });
}

/**
 * HDRnet-inspired architecture (Deep Bilateral Learning, SIGGRAPH 2017).
 *
 * The CNN consumes a 128×128 thumbnail and predicts a **bilateral grid** of
 * locally-affine colour transforms.  At full resolution the grid is sampled via
 * trilinear interpolation (in the Web Worker) and the resulting per-pixel
 * 3×4 affine matrix is applied to get the enhanced output.
 *
 * Grid dimensions:
 *   spatial  – 8 × 8   (matches the CNN's final spatial resolution)
 *   depth    – 8 bins   (intensity / luminance axis)
 *   coeffs   – 12 per cell  (3 output channels × (3 input + 1 bias))
 *
 * Architecture (lightweight, no BatchNorm):
 *
 *   Input 128×128×3
 *   ── Shared low-level features ──
 *   Conv(16, 3×3, stride 2) → ReLU   64²×16
 *   Conv(32, 3×3, stride 2) → ReLU   32²×32
 *   Conv(64, 3×3, stride 2) → ReLU   16²×64
 *   Conv(64, 3×3, stride 2) → ReLU    8²×64
 *
 *   ── Local pathway ──
 *   Conv(64, 3×3) → ReLU              8²×64
 *
 *   ── Global pathway ──
 *   Conv(64, 3×3, stride 2) → ReLU    4²×64
 *   GlobalAveragePooling2D             64
 *   Dense(64) → ReLU                  64
 *   Reshape → 1×1×64  →  UpSampling   8²×64
 *
 *   ── Fusion ──
 *   Add(local, global)                 8²×64
 *   Conv(96, 1×1, linear)             8²×96  →  reshape to 8×8×8×12
 */

const INPUT_SIZE = 128;
const GRID_W = 8;
const GRID_H = 8;
const GRID_D = 8;
const N_OUT = 3;
const N_IN_PLUS_BIAS = 4;
const N_COEFFS = N_OUT * N_IN_PLUS_BIAS; // 12
const GRID_SIZE = GRID_H * GRID_W * GRID_D * N_COEFFS; // 6144

export class ModelEngine {
  constructor() {
    this.model = null;
    this.ready = false;
    this.inputSize = INPUT_SIZE;
    this.gridW = GRID_W;
    this.gridH = GRID_H;
    this.gridD = GRID_D;
  }

  async init(onProgress) {
    const report = (stage) => {
      if (onProgress) onProgress(stage);
      return new Promise((r) => setTimeout(r, 0));
    };

    await report('loading');
    tf = await loadTf();

    await report('backend');
    try {
      await tf.setBackend('webgl');
      await tf.ready();
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
    }

    await report('build');
    this.model = this._buildModel();

    await report('weights');
    this._initGridOutputLayer();
    this._initFirstConvLayer();

    await report('warmup');
    tf.tidy(() => {
      const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
      this.model.predict(dummy);
    });

    this.ready = true;
  }

  _buildModel() {
    const input = tf.input({ shape: [INPUT_SIZE, INPUT_SIZE, 3] });

    let shared = this._convRelu(input, 16, 3, 2, 'shared1');
    shared      = this._convRelu(shared, 32, 3, 2, 'shared2');
    shared      = this._convRelu(shared, 64, 3, 2, 'shared3');
    shared      = this._convRelu(shared, 64, 3, 2, 'shared4');

    let local = this._convRelu(shared, 64, 3, 1, 'local1');

    let global = this._convRelu(shared, 64, 3, 2, 'global1');
    global      = tf.layers.globalAveragePooling2d({}).apply(global);
    global      = tf.layers.dense({ units: 64, activation: 'relu', name: 'global_fc' }).apply(global);
    global      = tf.layers.reshape({ targetShape: [1, 1, 64], name: 'global_reshape' }).apply(global);
    global      = tf.layers.upSampling2d({ size: [GRID_H, GRID_W], name: 'global_tile' }).apply(global);

    let fused = tf.layers.add({}).apply([local, global]);

    const gridOutput = tf.layers.conv2d({
      filters: GRID_D * N_COEFFS,
      kernelSize: 1,
      padding: 'same',
      name: 'grid_output',
    }).apply(fused);

    return tf.model({ inputs: input, outputs: gridOutput });
  }

  _convRelu(x, filters, kernelSize, strides, name) {
    x = tf.layers.conv2d({
      filters, kernelSize, strides, padding: 'same', name: `${name}_conv`,
    }).apply(x);
    x = tf.layers.activation({ activation: 'relu' }).apply(x);
    return x;
  }

  _initGridOutputLayer() {
    const layer = this.model.getLayer('grid_output');
    const [currentKernel] = layer.getWeights();

    const smallKernel = tf.mul(currentKernel, 0.001);

    const identity = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0];
    const biasData = new Float32Array(GRID_D * N_COEFFS);
    for (let d = 0; d < GRID_D; d++) {
      for (let c = 0; c < N_COEFFS; c++) {
        biasData[d * N_COEFFS + c] = identity[c];
      }
    }
    const newBias = tf.tensor1d(biasData);

    layer.setWeights([smallKernel, newBias]);
    smallKernel.dispose();
    newBias.dispose();
  }

  _initFirstConvLayer() {
    const layer = this.model.getLayer('shared1_conv');
    const [kernel] = layer.getWeights();
    const shape = kernel.shape; // [3, 3, 3, 16]
    const kData = Array.from(kernel.dataSync());
    const bData = new Float32Array(shape[3]).fill(0);
    const F = shape[3];

    const set = (h, w, ic, oc, v) => {
      kData[((h * shape[1] + w) * shape[2] + ic) * F + oc] = v;
    };

    for (let h = 0; h < 3; h++)
      for (let w = 0; w < 3; w++)
        for (let ic = 0; ic < 3; ic++)
          for (let oc = 0; oc < Math.min(8, F); oc++)
            set(h, w, ic, oc, 0);

    set(1, 1, 0, 0, 1); set(1, 1, 1, 1, 1); set(1, 1, 2, 2, 1);
    set(1, 1, 0, 3, 0.299); set(1, 1, 1, 3, 0.587); set(1, 1, 2, 3, 0.114);
    const lw = [0.299, 0.587, 0.114];
    for (let c = 0; c < 3; c++) {
      set(0, 0, c, 4, -lw[c]);  set(0, 2, c, 4,  lw[c]);
      set(1, 0, c, 4, -2*lw[c]);set(1, 2, c, 4,  2*lw[c]);
      set(2, 0, c, 4, -lw[c]);  set(2, 2, c, 4,  lw[c]);
    }
    for (let c = 0; c < 3; c++) {
      set(0, 0, c, 5, -lw[c]); set(0, 1, c, 5, -2*lw[c]); set(0, 2, c, 5, -lw[c]);
      set(2, 0, c, 5,  lw[c]); set(2, 1, c, 5,  2*lw[c]); set(2, 2, c, 5,  lw[c]);
    }
    for (let h = 0; h < 3; h++)
      for (let w = 0; w < 3; w++)
        for (let c = 0; c < 3; c++) set(h, w, c, 6, 1 / 9);
    for (let c = 0; c < 3; c++) {
      set(0, 1, c, 7, -1/3); set(1, 0, c, 7, -1/3);
      set(1, 1, c, 7,  4/3); set(1, 2, c, 7, -1/3);
      set(2, 1, c, 7, -1/3);
    }

    const newK = tf.tensor(kData, shape);
    const newB = tf.tensor1d(bData);
    layer.setWeights([newK, newB]);
    newK.dispose();
    newB.dispose();
  }

  predict(thumbnailData) {
    const gridTensor = tf.tidy(() => {
      const t = tf.browser.fromPixels(thumbnailData)
        .toFloat().div(255.0).expandDims(0);
      return this.model.predict(t);
    });
    const rawGrid = new Float32Array(gridTensor.dataSync());
    gridTensor.dispose();

    const features = this._extractFeatures(thumbnailData);
    const analytical = this._analyticalPredict(features);
    const analyticalAffine = this._buildAffine(analytical);

    const identity = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0];
    const delta = new Float32Array(N_COEFFS);
    for (let c = 0; c < N_COEFFS; c++) delta[c] = analyticalAffine[c] - identity[c];

    const grid = new Float32Array(rawGrid.length);
    for (let i = 0; i < rawGrid.length; i += N_COEFFS) {
      for (let c = 0; c < N_COEFFS; c++) {
        grid[i + c] = rawGrid[i + c] + delta[c];
      }
    }

    return {
      grid,
      gridW: GRID_W, gridH: GRID_H, gridD: GRID_D,
      analytical,
      features,
    };
  }

  _extractFeatures(imageData) {
    const d = imageData.data;
    const n = d.length / 4;
    let sR = 0, sG = 0, sB = 0, sBr = 0, sBr2 = 0, sSat = 0, dark = 0, light = 0;

    for (let i = 0; i < d.length; i += 4) {
      const r = d[i] / 255, g = d[i+1] / 255, b = d[i+2] / 255;
      sR += r; sG += g; sB += b;
      const br = 0.299*r + 0.587*g + 0.114*b;
      sBr += br; sBr2 += br*br;
      if (br < 0.25) dark++;
      if (br > 0.75) light++;
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      sSat += mx > 0 ? (mx - mn) / mx : 0;
    }

    const brightness = sBr / n;
    return {
      brightness,
      contrastStd: Math.sqrt(Math.max(0, sBr2 / n - brightness * brightness)),
      saturation: sSat / n,
      darkPct: dark / n,
      lightPct: light / n,
    };
  }

  _analyticalPredict(f) {
    let brightness = (0.45 - f.brightness) * 80;
    if (f.darkPct > 0.4) brightness += 15;
    if (f.lightPct > 0.4) brightness -= 15;
    brightness = Math.max(-60, Math.min(60, brightness));

    let contrast = 1.0;
    if (f.contrastStd < 0.12) contrast += (0.12 - f.contrastStd) * 4;
    else if (f.contrastStd > 0.25) contrast -= (f.contrastStd - 0.25) * 2;
    contrast = Math.max(0.5, Math.min(1.5, contrast));

    let saturation = 1.0;
    if (f.saturation < 0.25) saturation += (0.25 - f.saturation) * 3;
    else if (f.saturation > 0.55) saturation -= (f.saturation - 0.55) * 1.5;
    saturation = Math.max(0.5, Math.min(1.5, saturation));

    return {
      brightness: Math.round(brightness * 10) / 10,
      contrast:   Math.round(contrast * 1000) / 1000,
      saturation: Math.round(saturation * 1000) / 1000,
    };
  }

  _buildAffine({ brightness, contrast: c, saturation: s }) {
    const b = brightness / 255;
    const lr = 0.299, lg = 0.587, lb = 0.114;
    const bias = c * b + 0.5 * (1 - c);

    const affine = new Float32Array(12);
    const lum = [lr, lg, lb];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        affine[i * 4 + j] = c * (lum[j] * (1 - s) + (i === j ? s : 0));
      }
      affine[i * 4 + 3] = bias;
    }
    return affine;
  }

  async train(samples, { epochs = 50, batchSize = 4, onEpochEnd } = {}) {
    if (!this.model.optimizer) {
      this.model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
    }
    const xs = tf.tidy(() =>
      tf.stack(samples.map(s =>
        tf.browser.fromPixels(s.image).toFloat().div(255.0)
      ))
    );
    const ys = tf.tensor(
      samples.map(s => Array.from(s.gridTarget)),
      [samples.length, GRID_H, GRID_W, GRID_D * N_COEFFS],
    );

    const history = await this.model.fit(xs, ys, {
      epochs, batchSize, shuffle: true,
      callbacks: onEpochEnd ? { onEpochEnd } : undefined,
    });
    xs.dispose(); ys.dispose();
    return history;
  }

  async saveModel(path = 'downloads://hdrnet-model') {
    return this.model.save(path);
  }

  async loadModel(path) {
    this.model = await tf.loadLayersModel(path);
    this.ready = true;
  }

  dispose() {
    if (this.model) { this.model.dispose(); this.model = null; }
    this.ready = false;
  }
}
