const gallery = document.getElementById('gallery');

async function loadGallery() {
    if (!gallery) return;

    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    if (!isSafari) {
        const warning = document.createElement('div');
        warning.classList.add('warning');
        warning.textContent = 'Warning: This gallery contains HEIC images which are only supported in Safari.';
        gallery.parentElement?.insertBefore(warning, gallery);
    }

    try {
        const response = await fetch('images.json');
        if (!response.ok) {
            throw new Error('Failed to load image list');
        }
        const images: string[] = await response.json();

        gallery.innerHTML = '';

        if (images.length === 0) {
            gallery.innerHTML = '<p>No images found in public/images.</p>';
            return;
        }

        images.forEach(image => {
            const link = document.createElement('a');
            link.href = `images/${image}`;
            link.className = 'gallery-item';

            const img = document.createElement('img');
            img.src = `images/${image}`;
            img.alt = image;
            img.loading = 'lazy';

            link.appendChild(img);
            gallery.appendChild(link);
        });
    } catch (error) {
        gallery.innerHTML = `<p style="color:red">Error loading gallery: ${(error as Error).message}</p>`;
    }
}

loadGallery();