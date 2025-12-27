import { ModelRunner } from './inference';

const iiifUrlInput = document.getElementById('iiif-url') as HTMLInputElement;
const loadButton = document.getElementById('load-iiif') as HTMLButtonElement;
const appDiv = document.getElementById('app') as HTMLDivElement;

const runner = new ModelRunner('/model/vintagereality.onnx');

async function loadIIIFResource(url: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  }

  const data = await response.json();
  console.log('Loaded IIIF Data:', data);

  let imageUrl: string | undefined;

  // Check if it's an Image API info.json
  const isImageApi = data.protocol === 'http://iiif.io/api/image' || 
                     (data['@context'] && String(data['@context']).includes('image'));

  if (isImageApi) {
      const id = data['@id'] || data.id;
      if (id) imageUrl = `${id}/full/full/0/default.jpg`;
  } else {
      // Try Manifest V3
      const v3Image = data.items?.[0]?.items?.[0]?.items?.[0]?.body?.id;
      // Try Manifest V2
      const v2Image = data.sequences?.[0]?.canvases?.[0]?.images?.[0]?.resource?.['@id'];
      
      imageUrl = v3Image || v2Image;
  }
  return { imageUrl, data };
}

function preprocess(image: HTMLImageElement, width: number, height: number): Float32Array {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error("Canvas context unavailable");
    
    ctx.drawImage(image, 0, 0, width, height);
    const imgData = ctx.getImageData(0, 0, width, height);
    const { data } = imgData;
    
    // NCHW format: 1x3x640x640
    const float32Data = new Float32Array(3 * width * height);
    for (let i = 0; i < width * height; i++) {
        // Normalize to 0-1
        float32Data[i] = data[i * 4] / 255.0;                   // R
        float32Data[i + width * height] = data[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * width * height] = data[i * 4 + 2] / 255.0; // B
    }
    return float32Data;
}

function iou(box1: number[], box2: number[]) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    
    const b1_x1 = x1 - w1/2, b1_x2 = x1 + w1/2, b1_y1 = y1 - h1/2, b1_y2 = y1 + h1/2;
    const b2_x1 = x2 - w2/2, b2_x2 = x2 + w2/2, b2_y1 = y2 - h2/2, b2_y2 = y2 + h2/2;
    
    const inter_x1 = Math.max(b1_x1, b2_x1);
    const inter_y1 = Math.max(b1_y1, b2_y1);
    const inter_x2 = Math.min(b1_x2, b2_x2);
    const inter_y2 = Math.min(b1_y2, b2_y2);
    
    const inter_area = Math.max(0, inter_x2 - inter_x1) * Math.max(0, inter_y2 - inter_y1);
    const b1_area = w1 * h1;
    const b2_area = w2 * h2;
    
    return inter_area / (b1_area + b2_area - inter_area);
}

function postprocess(tensor: any, imgWidth: number, imgHeight: number, threshold: number = 0.25) {
    const output = tensor.data;
    const [batch, numFeatures, numAnchors] = tensor.dims; // e.g. 1, 5, 8400
    
    let boxes = [];
    
    // YOLOv8 output is [batch, features, anchors]
    // features: cx, cy, w, h, class_probs...
    
    for (let i = 0; i < numAnchors; i++) {
        // Find max class score
        let maxScore = 0;
        // Classes start at index 4
        for (let c = 4; c < numFeatures; c++) {
            const score = output[c * numAnchors + i];
            if (score > maxScore) maxScore = score;
        }
        
        if (maxScore > threshold) {
            const cx = output[0 * numAnchors + i];
            const cy = output[1 * numAnchors + i];
            const w = output[2 * numAnchors + i];
            const h = output[3 * numAnchors + i];
            
            boxes.push([cx, cy, w, h, maxScore]);
        }
    }
    
    // NMS
    boxes.sort((a, b) => b[4] - a[4]);
    const result = [];
    while (boxes.length > 0) {
        const best = boxes.shift();
        result.push(best);
        boxes = boxes.filter(b => iou(best, b) < 0.45);
    }
    
    // Scale to image size
    return result.map(box => {
        const [cx, cy, w, h, score] = box;
        return {
            x: (cx - w/2) / 640 * imgWidth,
            y: (cy - h/2) / 640 * imgHeight,
            w: w / 640 * imgWidth,
            h: h / 640 * imgHeight,
            score
        };
    });
}

loadButton?.addEventListener('click', async () => {
  const url = iiifUrlInput?.value;
  if (!url) {
    alert('Please enter a URL');
    return;
  }

  appDiv.innerHTML = 'Loading...';

  try {
    const { imageUrl, data } = await loadIIIFResource(url);

    if (imageUrl) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = imageUrl;
        img.style.maxWidth = "100%";

        appDiv.innerHTML = '';
        
        const container = document.createElement('div');
        container.style.position = 'relative';
        container.style.display = 'inline-block';
        container.appendChild(img);
        appDiv.appendChild(container);

        img.onload = async () => {
            try {
                const input = preprocess(img, 640, 640);
                const output = await runner.run(input);
                const boxes = postprocess(output as any, img.naturalWidth, img.naturalHeight);
                
                const canvas = document.createElement('canvas');
                canvas.style.position = 'absolute';
                canvas.style.left = '0';
                canvas.style.top = '0';
                canvas.style.width = '100%';
                canvas.style.height = '100%';
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                container.appendChild(canvas);
                
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 4;
                    boxes.forEach(box => ctx.strokeRect(box.x, box.y, box.w, box.h));
                }
                console.log('YOLO Inference Result:', boxes);
            } catch (e) {
                console.error("Inference failed:", e);
            }
        };
    } else {
        appDiv.innerHTML = `
          <h3>Loaded Successfully</h3>
          <pre style="overflow: auto; max-height: 80vh;">${JSON.stringify(data, null, 2)}</pre>
        `;
    }

  } catch (error) {
    console.error('Error loading IIIF:', error);
    appDiv.innerHTML = `<div style="color: red;">Error: ${(error as Error).message}</div>`;
  }
});