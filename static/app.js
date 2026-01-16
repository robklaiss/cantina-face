// Global constants/state
const FRAME_INTERVAL_MS = 500; // Keep in sync with backend config

let videoStream = null;
let recognitionInterval = null;
let recognitionEnabled = true;
let currentStudent = null;
let selectedProduct = null;
let products = [];
let searchTimeout = null;
let manualStudentSelection = false; // Track if user manually selected student
let fpsCounter = 0;
let lastFpsUpdate = Date.now();
let frameCount = 0;
let saleMode = false;
let cart = [];
let recognitionPausedByMode = false;

// Helpers
function formatGs(amount) {
    const n = Number(amount) || 0;
    try {
        return n.toLocaleString('es-PY', { maximumFractionDigits: 0 });
    } catch (_) {
        return n.toString();
    }
}

function navigateToSalesView(studentId) {
    if (!studentId) return;
    pauseRecognitionForMode(false);
    window.location.href = `/sales?student_id=${encodeURIComponent(studentId)}`;
}

function pauseRecognitionForMode(allowResume = true) {
    if (!recognitionEnabled) {
        recognitionPausedByMode = false;
        return;
    }
    recognitionPausedByMode = allowResume;
    recognitionEnabled = false;
    stopRecognition();
    stopCamera();
    recognitionIndicator.textContent = '⛔';
    recognitionIndicator.className = 'status-indicator warning';
    cameraIndicator.textContent = '❌';
    cameraIndicator.className = 'status-indicator error';
    if (toggleRecognitionBtn) {
        toggleRecognitionBtn.textContent = 'Encender reconocimiento';
    }
}

async function resumeRecognitionAfterMode() {
    if (!recognitionPausedByMode) {
        return;
    }
    recognitionPausedByMode = false;
    recognitionEnabled = true;
    if (toggleRecognitionBtn) {
        toggleRecognitionBtn.textContent = 'Apagar reconocimiento';
    }
    try {
        await startCamera();
        startRecognition();
        recognitionIndicator.textContent = '⏳';
        recognitionIndicator.className = 'status-indicator info';
    } catch (error) {
        console.error('Error resuming recognition:', error);
        showNotification('No se pudo reactivar la cámara automáticamente', 'error');
    }
}

// Search products (manual search in sale mode)
function searchProducts(query) {
    const trimmed = (query || '').trim().toLowerCase();
    searchResults.innerHTML = '';

    if (!trimmed) {
        return;
    }

    if (!products || products.length === 0) {
        searchResults.innerHTML = '<p class="no-data">Sin productos cargados</p>';
        return;
    }

    const matches = products.filter((product) => {
        const name = (product.name || '').toLowerCase();
        const priceStr = String(product.price || '');
        return name.includes(trimmed) || priceStr.includes(trimmed);
    });

    if (matches.length === 0) {
        searchResults.innerHTML = '<p class="no-data">Sin productos coincidentes</p>';
        return;
    }

    displayProductSearchResults(matches);
}

function displayProductSearchResults(productList) {
    searchResults.innerHTML = '';

    productList.forEach((product) => {
        const div = document.createElement('div');
        div.className = 'search-result';
        div.innerHTML = `
            <div class="search-info">
                <div class="search-name">${product.name}</div>
                <div class="search-grade">${formatGs(product.price)} Gs.</div>
                <div class="search-balance">${typeof product.stock === 'number' ? 'Stock: ' + product.stock : ''}</div>
            </div>
        `;

        div.addEventListener('click', () => {
            if (!currentStudent) {
                showNotification('No hay alumno seleccionado', 'error');
                return;
            }
            addToCart(product.id);
            searchInput.value = '';
            searchResults.innerHTML = '';
        });

        searchResults.appendChild(div);
    });
}

// DOM elements
const video = document.getElementById('video');
const studentCard = document.getElementById('student-card');
const noStudent = document.getElementById('no-student');
const studentZone = document.getElementById('student-zone');
const studentPhoto = document.getElementById('student-photo');
const studentName = document.getElementById('student-name');
const studentGrade = document.getElementById('student-grade');
const studentBalance = document.getElementById('student-balance');
const attachFaceBtn = document.getElementById('attach-face-btn');
const cameraIndicator = document.getElementById('camera-indicator');
const recognitionIndicator = document.getElementById('recognition-indicator');
const fpsValue = document.getElementById('fps-value');
const videoZone = document.getElementById('video-zone');
const productButtons = document.querySelectorAll('.product-btn');
const selectedProductSpan = document.getElementById('selected-product');
const selectedPriceSpan = document.getElementById('selected-price');
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const cartItems = document.getElementById('cart-items');
const cartTotal = document.getElementById('cart-total');
const productInfoPanel = document.getElementById('product-info');
const quickSearchInput = document.getElementById('quick-search-input');
const quickSearchResults = document.getElementById('quick-search-results');
const studentQuickSearch = document.getElementById('student-quick-search');
const enrollBtn = document.getElementById('enroll-btn');
const clearBtn = document.getElementById('clear-btn');
const chargeBtn = document.getElementById('charge-btn');
const manualCharge = document.getElementById('manual-charge');
const productsZone = document.getElementById('products-zone');
const searchZone = document.getElementById('search-zone');
const activeStudentName = document.getElementById('active-student-name');
const exitSaleBtn = document.getElementById('exit-sale-btn');
const toggleRecognitionBtn = document.getElementById('toggle-recognition-btn');
const adminBtn = document.getElementById('admin-btn');
const statusRight = document.querySelector('#status-bar .status-right');
let saleCloseBtn = null;

function ensureSaleCloseButton() {
    if (saleCloseBtn || !statusRight) return saleCloseBtn;
    saleCloseBtn = document.createElement('button');
    saleCloseBtn.id = 'sale-close-btn';
    saleCloseBtn.className = 'secondary-btn';
    saleCloseBtn.textContent = '✖️ Cerrar';
    saleCloseBtn.style.display = 'none';
    saleCloseBtn.addEventListener('click', () => clearBtn.click());
    statusRight.insertBefore(saleCloseBtn, toggleRecognitionBtn || null);
    return saleCloseBtn;
}

// Initialize application
async function init() {
    // Bind UI events early so navigation works even if initialization fails (e.g., camera denied)
    setupEventListeners();
    try {
        await loadProducts();
        await startCamera();
        startRecognition();
        showNotification('System initialized', 'success');
    } catch (error) {
        // Keep UI usable even on partial failures
        showNotification('Failed to initialize: ' + error.message, 'error');
    }

    // Attach current face button
    if (attachFaceBtn) {
        attachFaceBtn.addEventListener('click', attachCurrentFaceToStudent);
    }
}

// Load products from API
async function loadProducts() {
    try {
        const response = await fetch('/api/products');
        products = await response.json();

        // Update product buttons with names and prices and real IDs
        productButtons.forEach((btn, index) => {
            const product = products[index];
            if (product) {
                btn.dataset.productId = String(product.id);
                btn.title = `${product.name} — ${formatGs(product.price)} Gs.`;
                btn.innerHTML = `
                    <span class="kbd-index">${index + 1}</span>
                    <span class="label">${product.name}</span>
                    <span class="price">${formatGs(product.price)} Gs.</span>
                `;
                btn.disabled = false;
                btn.classList.remove('disabled');
            } else {
                // No product for this slot
                btn.removeAttribute('data-product-id');
                btn.innerHTML = `<span class="label">—</span>`;
                btn.title = '';
                btn.disabled = true;
                btn.classList.add('disabled');
            }
        });
    } catch (error) {
        showNotification('Failed to load products', 'error');
    }
}

// Start camera feed
async function startCamera() {
    console.log('📷 startCamera() called - requesting camera permission...');
    try {
        if (videoStream) {
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('Camera API not supported in this browser context.');
            throw new Error('Camera API not supported');
        }

        console.log('📹 Requesting camera stream...');
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });

        console.log('✅ Camera stream obtained');
        video.srcObject = videoStream;
        if (video.play) {
            await video.play().catch(() => {});
        }
        cameraIndicator.textContent = '✅';
        cameraIndicator.className = 'status-indicator success';
        console.log('📷 Camera initialized successfully');
    } catch (error) {
        console.error('❌ Camera error:', error);
        cameraIndicator.textContent = '❌';
        cameraIndicator.className = 'status-indicator error';
        throw new Error('Camera access denied: ' + (error && error.message ? error.message : String(error)));
    }
}

function stopCamera() {
    console.log('📷 stopCamera()');
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (video) {
        if (video.pause) {
            video.pause();
        }
        video.srcObject = null;
    }
    cameraIndicator.textContent = '❌';
    cameraIndicator.className = 'status-indicator error';
    console.log('📷 Camera stopped');
}

// Start face recognition loop
function startRecognition() {
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
    }

    if (!recognitionEnabled) {
        return;
    }

    lastFpsUpdate = Date.now();
    frameCount = 0;
    recognitionIndicator.textContent = '⏳';
    recognitionIndicator.className = 'status-indicator info';

    recognitionInterval = setInterval(async () => {
        if (!videoStream) return;
        if (!recognitionEnabled) {
            stopRecognition();
            return;
        }

        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            if (!blob) return; // Skip if blob creation failed
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            const response = await fetch('/api/recognize', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            updateRecognitionResult(result);

            // Update FPS counter
            frameCount++;
            const now = Date.now();
            if (now - lastFpsUpdate >= 1000) {
                fpsValue.textContent = frameCount;
                frameCount = 0;
                lastFpsUpdate = now;
            }

        } catch (error) {
            console.error('Recognition error:', error);
            recognitionIndicator.textContent = '❌';
        }
    }, FRAME_INTERVAL_MS);
}

function stopRecognition() {
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }

    frameCount = 0;
    recognitionIndicator.textContent = '⏹';
    recognitionIndicator.className = 'status-indicator';

    const faceBbox = document.getElementById('face-bbox');
    if (faceBbox) {
        faceBbox.style.display = 'none';
    }
}

// Update recognition result
function updateRecognitionResult(result) {
    // Update face detection visualization
    updateFaceDetectionOverlay(result.detection);

    // In sale mode with a manually selected student, keep current student fixed
    if (saleMode && manualStudentSelection) {
        return;
    }

    if (result.match) {
        showStudentCard(result.student, result.score);
    } else {
        // Only clear if user hasn't manually selected a student
        if (!manualStudentSelection) {
            currentStudent = null;
            hideStudentCard();
        }
        recognitionIndicator.textContent = '⏳';
    }
}

// Update face detection overlay
function updateFaceDetectionOverlay(detection) {
    const faceBbox = document.getElementById('face-bbox');
    const overlay = document.getElementById('video-overlay');

    if (!detection || !detection.face_detected || !detection.bbox) {
        faceBbox.style.display = 'none';
        return;
    }
    
    // Get overlay dimensions (coinciden visualmente con el video)
    const overlayRect = overlay.getBoundingClientRect();
    
    // Calculate bbox position relative to overlay
    const bbox = detection.bbox;
    const x1 = bbox.x1 * overlayRect.width;
    const y1 = bbox.y1 * overlayRect.height;
    const x2 = bbox.x2 * overlayRect.width;
    const y2 = bbox.y2 * overlayRect.height;
    
    const width = x2 - x1;
    const height = y2 - y1;
    
    // Update bbox element
    faceBbox.style.display = 'block';
    faceBbox.style.left = `${x1}px`;
    faceBbox.style.top = `${y1}px`;
    faceBbox.style.width = `${width}px`;
    faceBbox.style.height = `${height}px`;
    
    // Add confidence indicator
    const confidence = Math.round(detection.confidence * 100);
    faceBbox.innerHTML = `
        <div class="detection-info">
            <span class="confidence">Face: ${confidence}%</span>
        </div>
    `;
    
    // Color code based on confidence
    if (confidence > 80) {
        faceBbox.className = 'face-bbox high-confidence';
    } else if (confidence > 60) {
        faceBbox.className = 'face-bbox medium-confidence';
    } else {
        faceBbox.className = 'face-bbox low-confidence';
    }
}

// Show student card
function showStudentCard(student, score = null) {
    studentPhoto.src = student.photo_url || '/default-avatar.png';
    studentName.textContent = student.name;
    studentGrade.textContent = student.grade;
    studentBalance.textContent = formatGs(student.balance);

    currentStudent = student;

    if (activeStudentName) {
        activeStudentName.textContent = student.name;
    }

    studentCard.classList.remove('hidden');
    noStudent.classList.add('hidden');
    manualCharge.classList.remove('hidden');
}

// Hide student card
function hideStudentCard() {
    studentCard.classList.add('hidden');
    noStudent.classList.remove('hidden');
    manualCharge.classList.add('hidden');
    currentStudent = null;
    manualStudentSelection = false; // Reset manual selection flag

    if (activeStudentName && !saleMode) {
        activeStudentName.textContent = 'Ninguna';
    }
}

// Enter sale mode (show products and search)
function enterSaleMode() {
    if (!productsZone || !searchZone) {
        return;
    }

    saleMode = true;

    pauseRecognitionForMode();

    if (productsZone) {
        productsZone.classList.remove('hidden');
    }

    if (searchZone) {
        searchZone.classList.remove('hidden');
    }

    if (videoZone) {
        videoZone.classList.add('hidden');
    }

    if (studentQuickSearch) {
        studentQuickSearch.classList.add('hidden');
    }

    if (productInfoPanel) {
        productInfoPanel.classList.remove('hidden');
    }

    if (toggleRecognitionBtn) {
        toggleRecognitionBtn.style.display = 'none';
    }

    if (adminBtn) {
        adminBtn.style.display = 'none';
    }

    const closeBtn = ensureSaleCloseButton();
    if (closeBtn) {
        closeBtn.style.display = 'inline-flex';
    }

    if (searchInput) {
        setTimeout(() => searchInput.focus(), 100);
    }

    if (currentStudent && activeStudentName) {
        activeStudentName.textContent = currentStudent.name;
    }
}

// Cart helpers
function renderCart() {
    if (!cartItems || !cartTotal) return;

    cartItems.innerHTML = '';

    if (!cart || cart.length === 0) {
        cartItems.innerHTML = '<p class="no-data">Sin productos seleccionados</p>';
        cartTotal.textContent = '0';
        return;
    }

    let total = 0;

    cart.forEach(item => {
        const product = products.find(p => p.id === item.productId);
        if (!product) return;

        const subtotal = product.price * item.quantity;
        total += subtotal;

        const row = document.createElement('div');
        row.className = 'cart-item';
        const minusLabel = item.quantity > 1 ? '-' : '🗑️';
        const minusTitle = item.quantity > 1 ? 'Restar uno' : 'Eliminar producto';
        row.innerHTML = `
            <span class="cart-name">${product.name}</span>
            <div class="cart-qty">
                <span class="cart-subtotal">${formatGs(subtotal)} Gs.</span>
                <button class="qty-btn minus" title="${minusTitle}">${minusLabel}</button>
                <span class="qty">${item.quantity}</span>
                <button class="qty-btn plus" title="Sumar uno">+</button>
            </div>
        `;

        const minusBtn = row.querySelector('.qty-btn.minus');
        const plusBtn = row.querySelector('.qty-btn.plus');

        if (minusBtn) {
            minusBtn.addEventListener('click', () => updateCartQuantity(item.productId, -1));
        }
        if (plusBtn) {
            plusBtn.addEventListener('click', () => updateCartQuantity(item.productId, 1));
        }

        cartItems.appendChild(row);
    });

    cartTotal.textContent = formatGs(total);
}

function addToCart(productId) {
    const existing = cart.find(item => item.productId === productId);
    if (existing) {
        existing.quantity += 1;
    } else {
        cart.push({ productId, quantity: 1 });
    }
    renderCart();
}

// Attach current camera face to existing student
async function attachCurrentFaceToStudent() {
    if (!currentStudent) {
        showNotification('No hay alumno seleccionado', 'error');
        return;
    }
    if (!video || !video.videoWidth || !video.videoHeight) {
        showNotification('Cámara no disponible para capturar la cara', 'error');
        return;
    }

    try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
        if (!blob) {
            showNotification('No se pudo capturar la imagen de la cámara', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        const response = await fetch(`/api/students/${currentStudent.id}/attach-face`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            showNotification(result.detail || 'Error asociando la cara al alumno', 'error');
            return;
        }

        if (result.photo_url) {
            studentPhoto.src = result.photo_url;
        }

        showNotification('Cara actual asociada al alumno', 'success');
    } catch (error) {
        showNotification('Error asociando la cara: ' + error.message, 'error');
    }
}

function updateCartQuantity(productId, delta) {
    const item = cart.find(i => i.productId === productId);
    if (!item) return;

    item.quantity += delta;
    if (item.quantity <= 0) {
        cart = cart.filter(i => i.productId !== productId);
    }
    renderCart();
}

// Select product
function selectProduct(productId) {
    const product = products.find(p => p.id === productId);
    if (product) {
        selectedProduct = product;
        selectedProductSpan.textContent = product.name;
        selectedPriceSpan.textContent = product.price;

        // Update button appearance
        productButtons.forEach(btn => btn.classList.remove('selected'));
        const selectedBtn = document.querySelector(`[data-product-id="${productId}"]`);
        if (selectedBtn) {
            selectedBtn.classList.add('selected');
        }
    }
}

// Charge entire cart
async function chargeCart() {
    if (!currentStudent) {
        showNotification('No se seleccionó alumno', 'error');
        return;
    }

    if (!cart || cart.length === 0) {
        showNotification('No hay productos en el carrito', 'error');
        return;
    }

    // Pre-calcular total y validar saldo y stock usando datos locales
    let total = 0;
    for (const item of cart) {
        const product = products.find(p => p.id === item.productId);
        if (!product) continue;
        total += product.price * item.quantity;

        if (typeof product.stock === 'number' && product.stock < item.quantity) {
            showNotification(`Stock insuficiente para ${product.name}`, 'error');
            return;
        }
    }

    if (currentStudent.balance < total) {
        showNotification('Saldo insuficiente para todos los productos', 'error');
        return;
    }

    try {
        for (const item of cart) {
            const product = products.find(p => p.id === item.productId);
            if (!product) continue;

            for (let i = 0; i < item.quantity; i++) {
                const response = await fetch('/api/charge', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        student_id: currentStudent.id,
                        product_id: product.id
                    })
                });

                const result = await response.json();

                if (!response.ok) {
                    showNotification(result.detail || 'Error en el cobro', 'error');
                    return;
                }

                if (typeof result.new_balance === 'number') {
                    currentStudent.balance = result.new_balance;
                    studentBalance.textContent = formatGs(currentStudent.balance);
                }

                if (typeof result.new_stock === 'number') {
                    product.stock = result.new_stock;
                }
            }
        }

        // Todo ok: limpiar carrito y dar feedback
        cart = [];
        renderCart();

        studentCard.style.backgroundColor = '#e8f5e8';
        setTimeout(() => {
            studentCard.style.backgroundColor = '';
        }, 1000);

        showNotification(`✅ Compra cobrada a ${currentStudent.name}`, 'success');
    } catch (error) {
        showNotification('Error en el cobro: ' + error.message, 'error');
    }
}

// Search students
async function searchStudents(query) {
    const trimmed = (query || '').trim();
    if (!trimmed) {
        searchResults.innerHTML = '';
        return;
    }

    try {
        const response = await fetch(`/api/students?query=${encodeURIComponent(trimmed)}`);
        const students = await response.json();

        displaySearchResults(students);
    } catch (error) {
        console.error('Search error:', error);
    }
}

// Quick search students (top of student zone)
async function quickSearchStudents(query) {
    const trimmed = (query || '').trim();
    if (!trimmed) {
        quickSearchResults.innerHTML = '';
        return;
    }

    try {
        const response = await fetch(`/api/students?query=${encodeURIComponent(query)}`);
        const students = await response.json();

        displayQuickSearchResults(students);
    } catch (error) {
        console.error('Quick search error:', error);
    }
}

// Display search results
function displaySearchResults(students) {
    searchResults.innerHTML = '';

    students.forEach((student, index) => {
        const div = document.createElement('div');
        div.className = 'search-result';
        div.innerHTML = `
            <img src="${student.photo_url || '/default-avatar.png'}" alt="Foto" class="mini-photo" onerror="this.onerror=null;this.src='/default-avatar.png'">
            <div class="search-info">
                <div class="search-name">${student.name}</div>
                <div class="search-grade">${student.grade}</div>
                <div class="search-balance">${formatGs(student.balance)} Gs.</div>
            </div>
        `;

        div.addEventListener('click', () => {
            navigateToSalesView(student.id);
        });

        // Keyboard navigation
        div.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                div.click();
            } else if (e.key === 'ArrowDown' && index < students.length - 1) {
                e.preventDefault();
                searchResults.children[index + 1].focus();
            } else if (e.key === 'ArrowUp' && index > 0) {
                e.preventDefault();
                searchResults.children[index - 1].focus();
            }
        });

        searchResults.appendChild(div);
    });

    // Focus first result
    if (students.length > 0) {
        setTimeout(() => searchResults.children[0].focus(), 100);
    }
}

// Display quick search results (student zone)
function displayQuickSearchResults(students) {
    quickSearchResults.innerHTML = '';

    students.forEach((student) => {
        const div = document.createElement('div');
        div.className = 'search-result';
        div.innerHTML = `
            <img src="${student.photo_url || '/default-avatar.png'}" alt="Foto" class="mini-photo" onerror="this.onerror=null;this.src='/default-avatar.png'">
            <div class="search-info">
                <div class="search-name">${student.name}</div>
                <div class="search-grade">${student.grade}</div>
                <div class="search-balance">${student.balance} Gs.</div>
            </div>
        `;

        div.addEventListener('click', () => navigateToSalesView(student.id));

        quickSearchResults.appendChild(div);
    });
}

// Setup event listeners
function setupEventListeners() {
    // Student card: enter sale mode when clicking recognized/selected student
    if (studentCard) {
        studentCard.style.cursor = 'pointer';
        studentCard.addEventListener('click', () => {
            if (!currentStudent) return;
            navigateToSalesView(currentStudent.id);
        });
    }

    // Also allow clicking en la tarjeta de alumno dentro de la zona de alumno,
    // pero ignorar clicks en el buscador rápido para no forzar modo venta.
    if (studentZone) {
        studentZone.addEventListener('click', (e) => {
            if (!currentStudent) return;

            if (quickSearchInput && quickSearchInput.contains(e.target)) return;
            if (quickSearchResults && quickSearchResults.contains(e.target)) return;

            navigateToSalesView(currentStudent.id);
        });
    }

    // Product buttons
    productButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const productId = parseInt(btn.dataset.productId);
            if (!productId) return;
            if (!currentStudent) {
                showNotification('No se detectó alumno', 'error');
                return;
            }
            addToCart(productId);
        });
    });

    // Search input (manual search)
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            const value = e.target.value;
            searchTimeout = setTimeout(() => searchProducts(value), 150);
        });
    }

    // Quick search input (top of student zone)
    if (quickSearchInput) {
        quickSearchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => quickSearchStudents(e.target.value), 300);
        });
    }

    // Charge button
    if (chargeBtn) {
        chargeBtn.addEventListener('click', chargeCart);
    }

    // Exit sale button (Cerrar)
    if (exitSaleBtn) {
        exitSaleBtn.addEventListener('click', () => {
            clearBtn.click();
        });
    }

    // Clear button
    clearBtn.addEventListener('click', () => {
        hideStudentCard();
        selectedProduct = null;
        selectedProductSpan.textContent = '-';
        selectedPriceSpan.textContent = '0';
        productButtons.forEach(btn => btn.classList.remove('selected'));
        saleMode = false;

        resumeRecognitionAfterMode();

        if (productsZone) {
            productsZone.classList.add('hidden');
        }

        if (searchZone) {
            searchZone.classList.add('hidden');
        }

        if (videoZone) {
            videoZone.classList.remove('hidden');
        }

        if (studentQuickSearch) {
            studentQuickSearch.classList.remove('hidden');
        }

        if (productInfoPanel) {
            productInfoPanel.classList.add('hidden');
        }

        if (toggleRecognitionBtn) {
            toggleRecognitionBtn.style.display = '';
        }

        if (adminBtn) {
            adminBtn.style.display = '';
        }

        if (saleCloseBtn) {
            saleCloseBtn.style.display = 'none';
        }

        if (activeStudentName) {
            activeStudentName.textContent = 'Ninguna';
        }
    });

    // Enroll button
    if (enrollBtn) {
        enrollBtn.addEventListener('click', () => {
            window.location.href = '/enroll.html';
        });
    }

    if (adminBtn) {
        adminBtn.addEventListener('click', () => {
            pauseRecognitionForMode(false);
            window.location.href = '/admin.html';
        });
    }

    // Recognition toggle button
    if (toggleRecognitionBtn) {
        toggleRecognitionBtn.addEventListener('click', async () => {
            recognitionEnabled = !recognitionEnabled;

            if (recognitionEnabled) {
                toggleRecognitionBtn.textContent = 'Apagar reconocimiento';
                await startCamera();
                startRecognition();
                showNotification('Reconocimiento facial activado', 'info');
            } else {
                toggleRecognitionBtn.textContent = 'Encender reconocimiento';
                stopRecognition();
                stopCamera();
                recognitionIndicator.textContent = '⛔';
                recognitionIndicator.className = 'status-indicator warning';
                cameraIndicator.textContent = '❌';
                cameraIndicator.className = 'status-indicator error';
                showNotification('Reconocimiento facial desactivado', 'warning');
            }
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Number keys for products (1-9)
        if (e.key >= '1' && e.key <= '9') {
            e.preventDefault();
            const idx = parseInt(e.key, 10) - 1;
            if (products[idx]) {
                if (!currentStudent) {
                    showNotification('No se detectó alumno', 'error');
                    return;
                }
                addToCart(products[idx].id);
            }
            return;
        }

        // Enter key for charging
        if (e.key === 'Enter' && currentStudent && cart && cart.length > 0) {
            e.preventDefault();
            chargeCart();
            return;
        }

        // F2 for search focus
        if (e.key === 'F2') {
            e.preventDefault();
            if (!saleMode && productsZone && searchZone) {
                enterSaleMode();
            }

            if (productsZone && searchInput) {
                setTimeout(() => searchInput.focus(), 50);
            }
            return;
        }

        // Escape key for clear
        if (e.key === 'Escape') {
            e.preventDefault();
            clearBtn.click();
            return;
        }

        // Ctrl+E for enrollment
        if (e.key === 'e' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            window.location.href = '/enroll.html';
            return;
        }
    });

    // Focus search input on page load
    if (saleMode && searchInput) {
        searchInput.focus();
    }
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    const notifications = document.getElementById('notifications');
    notifications.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
