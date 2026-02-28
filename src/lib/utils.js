export function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substring(2, 8);
}

export async function loadImage(input, thumbSize = 256) {
  let blob = await toBlob(input);

  if (isHeicFormat(blob, input)) {
    const heic2any = (await import('heic2any')).default;
    const converted = await heic2any({ blob, toType: 'image/png' });
    blob = Array.isArray(converted) ? converted[0] : converted;
  }

  const bitmap = await createImageBitmap(blob);
  const { width, height } = bitmap;

  const fullCanvas = document.createElement('canvas');
  fullCanvas.width = width;
  fullCanvas.height = height;
  const fullCtx = fullCanvas.getContext('2d');
  fullCtx.drawImage(bitmap, 0, 0);
  const fullImageData = fullCtx.getImageData(0, 0, width, height);

  const thumbCanvas = document.createElement('canvas');
  thumbCanvas.width = thumbSize;
  thumbCanvas.height = thumbSize;
  const thumbCtx = thumbCanvas.getContext('2d');
  thumbCtx.drawImage(bitmap, 0, 0, thumbSize, thumbSize);
  const thumbnailData = thumbCtx.getImageData(0, 0, thumbSize, thumbSize);

  bitmap.close();

  return { fullImageData, thumbnailData, width, height };
}

export function imageDataToBlob(imageData, format = 'image/png', quality = 0.92) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    canvas.toBlob(resolve, format, quality);
  });
}

export function imageDataToObjectURL(imageData) {
  const canvas = document.createElement('canvas');
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

async function toBlob(input) {
  if (input instanceof Blob) return input;

  if (typeof input === 'string') {
    if (input.startsWith('data:')) {
      const res = await fetch(input);
      return res.blob();
    }
    const response = await fetch(input);
    return response.blob();
  }

  if (input instanceof HTMLImageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = input.naturalWidth;
    canvas.height = input.naturalHeight;
    canvas.getContext('2d').drawImage(input, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve));
  }

  if (input instanceof ImageData) {
    const canvas = document.createElement('canvas');
    canvas.width = input.width;
    canvas.height = input.height;
    canvas.getContext('2d').putImageData(input, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve));
  }

  throw new Error(
    'Unsupported input type. Provide a File, Blob, URL string, HTMLImageElement, or ImageData.'
  );
}

function isHeicFormat(blob, originalInput) {
  const type = blob.type?.toLowerCase() || '';
  if (type === 'image/heic' || type === 'image/heif') return true;
  if (originalInput instanceof File) {
    return /\.(heic|heif)$/i.test(originalInput.name);
  }
  if (typeof originalInput === 'string') {
    return /\.(heic|heif)(\?.*)?$/i.test(originalInput);
  }
  return false;
}
