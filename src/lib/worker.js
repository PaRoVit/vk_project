/**
 * Web Worker — bilateral grid slicing & affine application.
 *
 * Receives a flat bilateral grid (gridH × gridW × gridD × 12) produced by the
 * HDRnet CNN on the main thread, and applies it to every pixel of the full-
 * resolution image via trilinear interpolation in (x, y, luminance) space.
 *
 * Each grid cell stores a 3×4 affine matrix (12 floats):
 *   [ a00 a01 a02 a03 ]   R_out = a00·R + a01·G + a02·B + a03
 *   [ a10 a11 a12 a13 ]   G_out = a10·R + a11·G + a12·B + a13
 *   [ a20 a21 a22 a23 ]   B_out = a20·R + a21·G + a22·B + a23
 */

const cancelledTasks = new Set();

self.onmessage = function (e) {
  const { type, taskId } = e.data;
  if (type === 'cancel') { cancelledTasks.add(taskId); return; }
  if (type === 'process') processImage(e.data);
};

function processImage({ taskId, buffer, width, height, grid, gridW, gridH, gridD }) {
  const data = new Uint8ClampedArray(buffer);
  const totalPixels = width * height;
  const CHUNKS = 20;
  const pixelsPerChunk = Math.ceil(totalPixels / CHUNKS);
  const NC = 12; // coefficients per grid cell

  // Pre-compute grid-coordinate scale factors
  const scaleX = (gridW - 1) / Math.max(width  - 1, 1);
  const scaleY = (gridH - 1) / Math.max(height - 1, 1);
  const scaleZ = gridD - 1;

  let pixelsDone = 0;

  function processChunk() {
    if (cancelledTasks.has(taskId)) {
      cancelledTasks.delete(taskId);
      self.postMessage({ type: 'cancelled', taskId });
      return;
    }

    const end = Math.min(pixelsDone + pixelsPerChunk, totalPixels);

    for (let p = pixelsDone; p < end; p++) {
      const idx = p * 4;
      const r = data[idx]     / 255;
      const g = data[idx + 1] / 255;
      const b = data[idx + 2] / 255;

      // Guide signal – BT.709 luminance
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

      // Continuous grid coordinates
      const px = p % width;
      const py = (p - px) / width;
      const gx = px * scaleX;
      const gy = py * scaleY;
      const gz = Math.min(Math.max(lum, 0), 1) * scaleZ;

      // Integer corners (clamped)
      const gx0 = Math.min(gx | 0, gridW - 2);
      const gy0 = Math.min(gy | 0, gridH - 2);
      const gz0 = Math.min(gz | 0, gridD - 2);
      const gx1 = gx0 + 1;
      const gy1 = gy0 + 1;
      const gz1 = gz0 + 1;

      // Fractional parts
      const fx = gx - gx0, fy = gy - gy0, fz = gz - gz0;
      const ifx = 1 - fx, ify = 1 - fy, ifz = 1 - fz;

      // 8 trilinear weights
      const w000 = ifx * ify * ifz;
      const w001 = ifx * ify * fz;
      const w010 = fx  * ify * ifz;
      const w011 = fx  * ify * fz;
      const w100 = ifx * fy  * ifz;
      const w101 = ifx * fy  * fz;
      const w110 = fx  * fy  * ifz;
      const w111 = fx  * fy  * fz;

      // Base offsets into the flat grid array for 8 corners
      const b000 = ((gy0 * gridW + gx0) * gridD + gz0) * NC;
      const b001 = ((gy0 * gridW + gx0) * gridD + gz1) * NC;
      const b010 = ((gy0 * gridW + gx1) * gridD + gz0) * NC;
      const b011 = ((gy0 * gridW + gx1) * gridD + gz1) * NC;
      const b100 = ((gy1 * gridW + gx0) * gridD + gz0) * NC;
      const b101 = ((gy1 * gridW + gx0) * gridD + gz1) * NC;
      const b110 = ((gy1 * gridW + gx1) * gridD + gz0) * NC;
      const b111 = ((gy1 * gridW + gx1) * gridD + gz1) * NC;

      // Interpolate 12 affine coefficients and apply in one pass
      let rOut = 0, gOut = 0, bOut = 0;
      const inp = [r, g, b, 1];

      for (let ch = 0; ch < 3; ch++) {
        const co = ch * 4;         // offset within the 12 coefficients
        let val = 0;
        for (let j = 0; j < 4; j++) {
          const ci = co + j;
          val += (
            grid[b000+ci]*w000 + grid[b001+ci]*w001 +
            grid[b010+ci]*w010 + grid[b011+ci]*w011 +
            grid[b100+ci]*w100 + grid[b101+ci]*w101 +
            grid[b110+ci]*w110 + grid[b111+ci]*w111
          ) * inp[j];
        }
        if (ch === 0) rOut = val;
        else if (ch === 1) gOut = val;
        else bOut = val;
      }

      data[idx]     = rOut * 255;
      data[idx + 1] = gOut * 255;
      data[idx + 2] = bOut * 255;
      // alpha untouched
    }

    pixelsDone = end;
    const progress = Math.round((pixelsDone / totalPixels) * 100);
    self.postMessage({ type: 'progress', taskId, progress: Math.min(progress, 100) });

    if (pixelsDone < totalPixels) {
      setTimeout(processChunk, 0);
    } else {
      self.postMessage(
        { type: 'complete', taskId, buffer: data.buffer, width, height },
        [data.buffer],
      );
    }
  }

  processChunk();
}
