const iiifUrlInput = document.getElementById('iiif-url') as HTMLInputElement;
const loadButton = document.getElementById('load-iiif') as HTMLButtonElement;
const appDiv = document.getElementById('app') as HTMLDivElement;

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
        appDiv.innerHTML = `<img src="${imageUrl}" style="max-width: 100%;" />`;
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