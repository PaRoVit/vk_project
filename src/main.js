import './styles.css';
import { ImageEnhancer } from './lib/index.js';

const enhancer = new ImageEnhancer();

const $ = (sel) => document.querySelector(sel);
const overlay = $('.init-overlay');
const dropzone = $('.dropzone');
const fileInput = $('#fileInput');
const statusBar = $('.status-bar');
const statusLabel = $('.status-value');
const progressFill = $('.progress-fill');
const cancelBtn = $('.cancel-btn');
const paramsCard = $('.params-card');
const paramBrightness = $('#paramBrightness');
const paramContrast = $('#paramContrast');
const paramSaturation = $('#paramSaturation');
const comparison = $('.comparison');
const originalImg = $('#originalImg');
const enhancedImg = $('#enhancedImg');
const downloadBtn = $('.download-btn');

let currentTaskId = null;
let originalObjectURL = null;
let enhancedBlobURL = null;

async function initApp() {
  try {
    await enhancer.init();
    overlay.classList.add('hidden');
  } catch (err) {
    overlay.querySelector('p').textContent = `Initialization failed: ${err.message}`;
    console.error(err);
  }
}

enhancer.on('statusChange', (detail) => {
  if (detail.taskId !== currentTaskId) return;

  statusBar.classList.add('visible');
  progressFill.classList.remove('done', 'error');

  const statusText = {
    pending: 'Waiting...',
    loading: 'Loading image...',
    analyzing: 'HDRnet is predicting bilateral grid...',
    processing: `Processing... ${detail.progress}%`,
    completed: 'Done!',
    cancelled: 'Cancelled',
    error: `Error: ${detail.error || 'unknown'}`,
  };

  statusLabel.textContent = statusText[detail.status] || detail.status;

  switch (detail.status) {
    case 'loading':
    case 'analyzing':
      progressFill.style.width = '0%';
      cancelBtn.style.display = 'inline-block';
      break;
    case 'processing':
      progressFill.style.width = `${detail.progress}%`;
      break;
    case 'completed':
      progressFill.style.width = '100%';
      progressFill.classList.add('done');
      cancelBtn.style.display = 'none';
      showParams(detail.params);
      showResult();
      break;
    case 'cancelled':
      progressFill.style.width = '0%';
      cancelBtn.style.display = 'none';
      break;
    case 'error':
      progressFill.classList.add('error');
      progressFill.style.width = '100%';
      cancelBtn.style.display = 'none';
      break;
  }
});

function showParams(params) {
  if (!params) return;
  paramsCard.classList.add('visible');
  const sign = (v) => (v >= 0 ? '+' : '');
  paramBrightness.textContent = `${sign(params.brightness)}${params.brightness}`;
  paramContrast.textContent = `x${params.contrast}`;
  paramSaturation.textContent = `x${params.saturation}`;
}

async function showResult() {
  try {
    const blob = await enhancer.getResult(currentTaskId);
    if (enhancedBlobURL) URL.revokeObjectURL(enhancedBlobURL);
    enhancedBlobURL = URL.createObjectURL(blob);
    enhancedImg.src = enhancedBlobURL;
    comparison.classList.add('visible');
  } catch (err) {
    console.error('Failed to get result:', err);
  }
}

async function handleFile(file) {
  if (!file) return;

  resetUI();

  originalObjectURL = URL.createObjectURL(file);
  originalImg.src = originalObjectURL;

  try {
    currentTaskId = await enhancer.submitTask(file);
  } catch (err) {
    statusBar.classList.add('visible');
    statusLabel.textContent = `Error: ${err.message}`;
    progressFill.classList.add('error');
    progressFill.style.width = '100%';
  }
}

function resetUI() {
  if (originalObjectURL) URL.revokeObjectURL(originalObjectURL);
  if (enhancedBlobURL) URL.revokeObjectURL(enhancedBlobURL);
  originalObjectURL = null;
  enhancedBlobURL = null;

  statusBar.classList.remove('visible');
  paramsCard.classList.remove('visible');
  comparison.classList.remove('visible');
  progressFill.style.width = '0%';
  progressFill.classList.remove('done', 'error');
  cancelBtn.style.display = 'inline-block';
  statusLabel.textContent = '';
}

dropzone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('active');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('active');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('active');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

cancelBtn.addEventListener('click', () => {
  if (currentTaskId) enhancer.cancelTask(currentTaskId);
});

downloadBtn.addEventListener('click', async () => {
  if (!currentTaskId) return;
  try {
    const blob = await enhancer.getResult(currentTaskId, {
      mimeType: 'image/png',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'enhanced-image.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Download failed:', err);
  }
});

initApp();
