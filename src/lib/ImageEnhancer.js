import { ModelEngine } from './ModelEngine.js';
import { generateId, loadImage, imageDataToBlob, imageDataToObjectURL } from './utils.js';

const STATUS = Object.freeze({
  PENDING: 'pending',
  LOADING: 'loading',
  ANALYZING: 'analyzing',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  CANCELLED: 'cancelled',
  ERROR: 'error',
});

export class ImageEnhancer {
  constructor() {
    this._model = new ModelEngine();
    this._worker = null;
    this._tasks = new Map();
    this._listeners = new Map();
    this._initialized = false;
  }

  async init() {
    if (this._initialized) return;
    await this._model.init();
    this._worker = new Worker(new URL('./worker.js', import.meta.url));
    this._worker.onmessage = (e) => this._handleWorkerMessage(e);
    this._worker.onerror = (e) => this._handleWorkerError(e);
    this._initialized = true;
  }

  /**
   * Submit an image for AI-powered enhancement.
   * Accepts: File, Blob, URL string, HTMLImageElement, or ImageData.
   * Returns a task ID for tracking progress.
   */
  async submitTask(input) {
    if (!this._initialized) {
      throw new Error('ImageEnhancer is not initialized. Call init() first.');
    }

    const taskId = generateId();
    this._tasks.set(taskId, {
      id: taskId,
      status: STATUS.PENDING,
      progress: 0,
      params: null,
      error: null,
      _resultBuffer: null,
      _width: 0,
      _height: 0,
    });

    this._emitStatus(taskId, STATUS.PENDING, 0);
    this._processTask(taskId, input);
    return taskId;
  }

  getTaskStatus(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) throw new Error(`Task ${taskId} not found`);
    return {
      taskId: task.id,
      status: task.status,
      progress: task.progress,
      params: task.params,
      error: task.error,
    };
  }

  cancelTask(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) throw new Error(`Task ${taskId} not found`);

    if (task.status === STATUS.COMPLETED || task.status === STATUS.CANCELLED) {
      return { success: false, reason: `Task is already ${task.status}` };
    }
    if (this._worker && task.status === STATUS.PROCESSING) {
      this._worker.postMessage({ type: 'cancel', taskId });
    }
    task.status = STATUS.CANCELLED;
    task.progress = 0;
    this._emitStatus(taskId, STATUS.CANCELLED, 0);
    return { success: true };
  }

  async getResult(taskId, options = {}) {
    const task = this._tasks.get(taskId);
    if (!task) throw new Error(`Task ${taskId} not found`);
    if (task.status !== STATUS.COMPLETED) {
      throw new Error(`Task ${taskId} is not completed (status: ${task.status})`);
    }

    const { format = 'blob', mimeType = 'image/png', quality = 0.92 } = options;
    const imageData = new ImageData(
      new Uint8ClampedArray(task._resultBuffer),
      task._width,
      task._height,
    );

    switch (format) {
      case 'blob':    return imageDataToBlob(imageData, mimeType, quality);
      case 'dataurl': return imageDataToObjectURL(imageData);
      case 'imagedata': return imageData;
      default:        return imageDataToBlob(imageData, mimeType, quality);
    }
  }

  on(event, callback) {
    if (!this._listeners.has(event)) this._listeners.set(event, new Set());
    this._listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    const set = this._listeners.get(event);
    if (set) set.delete(callback);
  }

  destroy() {
    if (this._worker) { this._worker.terminate(); this._worker = null; }
    this._model.dispose();
    this._tasks.clear();
    this._listeners.clear();
    this._initialized = false;
  }

  // ---- internal ----

  async _processTask(taskId, input) {
    const task = this._tasks.get(taskId);
    if (!task) return;

    try {
      // 1. Load image
      task.status = STATUS.LOADING;
      this._emitStatus(taskId, STATUS.LOADING, 0);

      const thumbSize = this._model.inputSize; // 256
      const { fullImageData, thumbnailData, width, height } = await loadImage(input, thumbSize);
      if (task.status === STATUS.CANCELLED) return;

      // 2. CNN predicts bilateral grid + analytical baseline
      task.status = STATUS.ANALYZING;
      this._emitStatus(taskId, STATUS.ANALYZING, 0);

      const prediction = this._model.predict(thumbnailData);
      task.params = {
        brightness: prediction.analytical.brightness,
        contrast:   prediction.analytical.contrast,
        saturation: prediction.analytical.saturation,
      };
      if (task.status === STATUS.CANCELLED) return;

      // 3. Send full image + composed grid to worker
      task.status = STATUS.PROCESSING;
      task._width = width;
      task._height = height;
      this._emitStatus(taskId, STATUS.PROCESSING, 0);

      const { grid, gridW, gridH, gridD } = prediction;
      const imgBuffer = fullImageData.data.buffer;
      this._worker.postMessage(
        { type: 'process', taskId, buffer: imgBuffer, width, height,
          grid, gridW, gridH, gridD },
        [imgBuffer, grid.buffer],
      );
    } catch (error) {
      console.error(`Task ${taskId} failed:`, error);
      task.status = STATUS.ERROR;
      task.error = error.message;
      this._emitStatus(taskId, STATUS.ERROR, 0);
    }
  }

  _handleWorkerMessage(e) {
    const { type, taskId, progress, buffer, width, height } = e.data;
    const task = this._tasks.get(taskId);
    if (!task) return;

    switch (type) {
      case 'progress':
        task.progress = progress;
        this._emitStatus(taskId, STATUS.PROCESSING, progress);
        break;
      case 'complete':
        task.status = STATUS.COMPLETED;
        task.progress = 100;
        task._resultBuffer = buffer;
        task._width = width;
        task._height = height;
        this._emitStatus(taskId, STATUS.COMPLETED, 100);
        break;
      case 'cancelled':
        task.status = STATUS.CANCELLED;
        this._emitStatus(taskId, STATUS.CANCELLED, task.progress);
        break;
      case 'error':
        task.status = STATUS.ERROR;
        task.error = e.data.error;
        this._emitStatus(taskId, STATUS.ERROR, task.progress);
        break;
    }
  }

  _handleWorkerError(e) {
    console.error('Worker error:', e);
  }

  _emitStatus(taskId, status, progress) {
    const task = this._tasks.get(taskId);
    const detail = {
      taskId, status, progress,
      params: task?.params || null,
      error: task?.error || null,
    };
    const set = this._listeners.get('statusChange');
    if (set) {
      set.forEach((cb) => {
        try { cb(detail); } catch (err) { console.error('Event callback error:', err); }
      });
    }
  }
}

export { STATUS };
