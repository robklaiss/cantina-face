// Global variables for admin panel
let currentStudents = [];
let currentProducts = [];
let currentStudent = null;
let adminVideoStream = null;
let currentSection = 'students';
let searchDebounceTimer = null;
let productSearchDebounceTimer = null;
const DEBOUNCE_DELAY = 300;
let topProductsChartInstance = null;
let topStudentsChartInstance = null;
let dailySalesChartInstance = null;
const AUTH_TOKEN_KEY = 'cantina_face_token';
const AUTH_USER_KEY = 'cantina_face_user';
let adminAuthToken = localStorage.getItem(AUTH_TOKEN_KEY) || null;
let adminAuthEmail = localStorage.getItem(AUTH_USER_KEY) || null;
const nativeFetch = window.fetch.bind(window);

function formatGs(amount) {
    const n = Number(amount) || 0;
    try {
        return n.toLocaleString('es-PY', { maximumFractionDigits: 0 });
    } catch (_) {
        return n.toString();
    }
}

// DOM elements (assigned at init time)
let adminSidebar, adminContent, studentModal, studentsGrid, studentSearch, productSearchInput, adminLogoutBtn;

function currentPathWithFallback() {
    const path = `${window.location.pathname}${window.location.search}`;
    if (!path || path === '/' || path.startsWith('/login')) {
        return '/admin.html';
    }
    return path;
}

function redirectToLogin() {
    const next = encodeURIComponent(currentPathWithFallback());
    window.location.href = `/login.html?next=${next}`;
}

function setAuthSession(token, email) {
    adminAuthToken = token || null;
    adminAuthEmail = email || null;
    if (adminAuthToken) {
        localStorage.setItem(AUTH_TOKEN_KEY, adminAuthToken);
        if (adminAuthEmail) {
            localStorage.setItem(AUTH_USER_KEY, adminAuthEmail);
        }
    } else {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
    }
}

function handleUnauthorized() {
    showNotification('Sesión expirada, vuelve a ingresar', 'error');
    logoutAdmin(true);
}

async function fetchWithAuth(url, options = {}) {
    const headers = options.headers ? { ...options.headers } : {};
    if (adminAuthToken && !headers['Authorization']) {
        headers['Authorization'] = `Bearer ${adminAuthToken}`;
    }
    const response = await nativeFetch(url, { ...options, headers });
    if (response.status === 401) {
        handleUnauthorized();
    }
    return response;
}

window.fetch = fetchWithAuth;

function ensureAuthenticated() {
    if (!adminAuthToken) {
        redirectToLogin();
        return false;
    }
    return true;
}

function logoutAdmin(silent = false) {
    setAuthSession(null, null);
    if (!silent) {
        showNotification('Sesión cerrada', 'info');
    }
    redirectToLogin();
}

// Initialize admin panel
function initAdminPanel() {
    if (!ensureAuthenticated()) {
        return;
    }
    console.log('[admin] initAdminPanel');
    // Assign DOM elements now that DOM is ready
    adminSidebar = document.getElementById('admin-sidebar');
    adminContent = document.getElementById('admin-content');
    studentModal = document.getElementById('student-modal');
    studentsGrid = document.getElementById('students-grid');
    studentSearch = document.getElementById('student-search');
    productSearchInput = document.getElementById('product-search');
    adminLogoutBtn = document.getElementById('admin-logout-btn');
    setupAdminEventListeners();
    setupNavigation();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAdminPanel);
} else {
    // DOM already parsed
    initAdminPanel();
}

// Setup event listeners
function setupAdminEventListeners() {
    console.log('[admin] setupAdminEventListeners');
    // Navigation
    const backBtn = document.getElementById('back-to-main');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            window.location.href = '/index.html';
        });
    }

    if (adminLogoutBtn) {
        adminLogoutBtn.addEventListener('click', () => logoutAdmin());
    }

    // Section navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchSection(e.target.dataset.section);
        });
    });

    // Student search
    if (studentSearch) {
        studentSearch.addEventListener('input', (e) => {
            filterStudents(e.target.value);
        });
    }

    if (productSearchInput) {
        productSearchInput.addEventListener('input', (e) => {
            filterProducts(e.target.value);
        });
    }

    // Add student button
    const addStudentBtn = document.getElementById('add-student-btn');
    if (addStudentBtn) {
        addStudentBtn.addEventListener('click', () => {
            window.location.href = '/enroll.html';
        });
    }

    // Add product button
    const addProductBtn = document.getElementById('add-product-btn');
    if (addProductBtn) {
        addProductBtn.addEventListener('click', showAddProductModal);
    }

    const transactionsRunBtn = document.getElementById('transactions-run-btn');
    if (transactionsRunBtn) {
        transactionsRunBtn.addEventListener('click', loadTransactions);
    }

    // Modal controls
    document.querySelectorAll('.close-btn').forEach(btn => {
        btn.addEventListener('click', closeStudentModal);
    });

    // Close on overlay click
    if (studentModal) {
        studentModal.addEventListener('click', (e) => {
            if (e.target === studentModal) closeStudentModal();
        });
    }

    // Close on Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && studentModal && !studentModal.classList.contains('hidden')) {
            closeStudentModal();
        }
    });

    // Photo controls - check if elements exist before adding listeners
    const retakeBtn = document.getElementById('retake-photo-btn');
    const addPhotoBtn = document.getElementById('add-photo-btn');
    const captureBtn = document.getElementById('capture-btn');
    const cancelCaptureBtn = document.getElementById('cancel-capture-btn');
    const addCreditsBtn = document.getElementById('add-credits-btn');
    const saveStudentBtn = document.getElementById('save-student-btn');
    const deleteStudentBtn = document.getElementById('delete-student-btn');

    if (retakeBtn) retakeBtn.addEventListener('click', startPhotoCapture);
    if (addPhotoBtn) addPhotoBtn.addEventListener('click', startPhotoCapture);
    if (captureBtn) captureBtn.addEventListener('click', capturePhoto);
    if (cancelCaptureBtn) cancelCaptureBtn.addEventListener('click', cancelPhotoCapture);
    if (addCreditsBtn) addCreditsBtn.addEventListener('click', addCredits);
    if (saveStudentBtn) saveStudentBtn.addEventListener('click', saveStudent);
    if (deleteStudentBtn) deleteStudentBtn.addEventListener('click', () => showDeleteConfirmation());

    // Delegate clicks from students grid for dynamic buttons
    if (studentsGrid) {
        studentsGrid.addEventListener('click', (e) => {
            const card = e.target.closest('.student-card');
            if (!card) return;
            const id = card.dataset.id;
            if (!id) return;
            if (e.target.closest('.edit-btn')) {
                editStudent(id);
            } else if (e.target.closest('.history-btn')) {
                viewHistory(id);
            }
        });
    }
}

// Switch between admin sections
function switchSection(section) {
    // Prevent unnecessary reloading of same section
    if (currentSection === section) return;
    
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.querySelector(`[data-section="${section}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    // Update content
    document.querySelectorAll('.admin-section').forEach(sec => {
        sec.classList.remove('active');
    });
    const activeSection = document.getElementById(`${section}-section`);
    if (activeSection) activeSection.classList.add('active');

    currentSection = section;

    // Load section-specific data only once
    switch(section) {
        case 'students':
            if (currentStudents.length === 0) loadStudents();
            break;
        case 'products':
            loadProducts();
            break;
        case 'transactions':
            loadTransactions();
            break;
        case 'analytics':
            loadAnalytics();
            break;
    }
}

// Load students
async function loadStudents() {
    try {
        const response = await fetchWithAuth('/api/students?limit=100');
        const students = await response.json();
        currentStudents = students;
        renderStudents(students);
        populateTransactionFilter();
    } catch (error) {
        showNotification('Error loading students: ' + error.message, 'error');
    }
}

// Render students grid
function renderStudents(students) {
    studentsGrid.innerHTML = '';
    console.log('[admin] renderStudents count=', students ? students.length : 0);
    
    if (!students || students.length === 0) {
        studentsGrid.innerHTML = '<p class="no-data">No students found</p>';
        return;
    }
    
    const fragment = document.createDocumentFragment();
    students.forEach(student => {
        const studentCard = document.createElement('div');
        studentCard.className = 'student-card';
        // store id for delegation
        try { studentCard.dataset.id = String(student.id); } catch(_) {}
        
        // Create photo container with loading state
        const photoDiv = document.createElement('div');
        photoDiv.className = 'student-photo';
        
        // Create image with explicit dimensions and loading="eager"
        const img = document.createElement('img');
        img.alt = student.name;
        img.width = 80;
        img.height = 80;
        img.loading = 'eager';
        img.style.background = '#2d2d2d';
        img.onerror = () => {
            img.onerror = null;
            img.src = '/default-avatar.png';
        };
        
        // Set source after setting up handlers
        img.src = student.photo_url || '/default-avatar.png';
        
        photoDiv.appendChild(img);
        
        studentCard.innerHTML = `
            <div class="student-info">
                <h4>${student.name}</h4>
                <p class="grade">${student.grade}</p>
                <p class="balance">Saldo: ${formatGs(student.balance)} Gs.</p>
                <p class="student-id-row">ID:
                    <code>${student.id}</code>
                    <button type="button" class="copy-id-btn" data-id="${student.id}">Copiar ID</button>
                </p>
            </div>
            <div class="student-actions">
                <button class="edit-btn">✏️ Editar</button>
                <button class="history-btn">📊 Historial</button>
            </div>
        `;
        
        // Insert photo at beginning
        studentCard.insertBefore(photoDiv, studentCard.firstChild);

        // Wire up actions programmatically (direct binding)
        const editBtn = studentCard.querySelector('.edit-btn');
        const historyBtn = studentCard.querySelector('.history-btn');
        if (editBtn) editBtn.addEventListener('click', () => editStudent(student.id));
        if (historyBtn) historyBtn.addEventListener('click', () => viewHistory(student.id));
        const copyBtn = studentCard.querySelector('.copy-id-btn');
        if (copyBtn) {
            copyBtn.addEventListener('click', (event) => {
                event.stopPropagation();
                const value = copyBtn.dataset.id || '';
                if (!value) {
                    showNotification('ID no disponible', 'error');
                    return;
                }
                navigator.clipboard?.writeText(value)
                    .then(() => showNotification('ID de alumno copiado', 'success'))
                    .catch(() => showNotification('No se pudo copiar el ID', 'error'));
            });
        }
        fragment.appendChild(studentCard);
    });
    
    studentsGrid.appendChild(fragment);
}

// Filter students
function filterStudents(query) {
  clearTimeout(searchDebounceTimer);
  
  // Early exit if query is empty
  if (!query.trim()) {
    renderStudents(currentStudents);
    return;
  }

  searchDebounceTimer = setTimeout(() => {
    const filtered = currentStudents.filter(student => 
      student.name.toLowerCase().includes(query.toLowerCase()) ||
      student.grade.toLowerCase().includes(query.toLowerCase())
    );
    renderStudents(filtered);
  }, DEBOUNCE_DELAY);
}

function populateTransactionFilter() {
    const select = document.getElementById('transaction-filter');
    if (!select) return;

    const previousValue = select.value;

    select.innerHTML = '<option value="">Todos los estudiantes</option>';

    if (!currentStudents || currentStudents.length === 0) return;

    currentStudents.forEach(student => {
        const option = document.createElement('option');
        option.value = String(student.id);
        option.textContent = `${student.name} (${student.grade})`;
        select.appendChild(option);
    });

    if (previousValue && select.querySelector(`option[value="${previousValue}"]`)) {
        select.value = previousValue;
    }
}

// Edit student
async function editStudent(studentId) {
    try {
        const idStr = String(studentId);
        const student = currentStudents.find(s => String(s.id) === idStr);
        if (!student) return;

        currentStudent = student;
        
        // Populate form
        document.getElementById('edit-name').value = student.name;
        document.getElementById('edit-grade').value = student.grade;
        document.getElementById('edit-balance').value = student.balance;
        document.getElementById('current-student-photo').src = student.photo_url || '/default-avatar.png';
        const modalTitle = document.getElementById('modal-title');
        if (modalTitle) modalTitle.textContent = 'Editar Estudiante';
        
        // Show modal first for responsiveness (force display in case of CSS conflicts)
        studentModal.classList.remove('hidden');
        studentModal.style.display = 'flex';
        
        // Load purchase history and suggestions asynchronously
        loadStudentPurchases(student.id);
        loadSuggestedProducts(student.id);
        
    } catch (error) {
        showNotification('Error loading student: ' + error.message, 'error');
    }
}

// Load student purchase history
async function loadStudentPurchases(studentId) {
    try {
        const response = await fetch(`/api/students/${studentId}/transactions`);
        const transactions = await response.json();
        
        const purchasesContainer = document.getElementById('student-purchases');
        purchasesContainer.innerHTML = '';
        
        if (transactions.length === 0) {
            purchasesContainer.innerHTML = '<p class="no-data">Sin compras registradas</p>';
            return;
        }
        
        transactions.forEach(transaction => {
            const transactionEl = document.createElement('div');
            transactionEl.className = 'transaction-item';
            transactionEl.innerHTML = `
                <div class="transaction-info">
                    <span class="product-name">${transaction.product_name}</span>
                    <span class="transaction-date">${new Date(transaction.created_at).toLocaleDateString()}</span>
                </div>
                <div class="transaction-amount">-${formatGs(transaction.amount)} Gs.</div>
            `;
            purchasesContainer.appendChild(transactionEl);
        });
        
    } catch (error) {
        document.getElementById('student-purchases').innerHTML = '<p class="error">Error cargando historial</p>';
    }
}

// Load suggested products based on purchase history
async function loadSuggestedProducts(studentId) {
    try {
        const response = await fetch(`/api/students/${studentId}/suggestions`);
        const suggestions = await response.json();
        
        const suggestionsContainer = document.getElementById('suggested-products-list');
        suggestionsContainer.innerHTML = '';
        
        if (suggestions.length === 0) {
            suggestionsContainer.innerHTML = '<p class="no-data">Sin sugerencias disponibles</p>';
            return;
        }
        
        suggestions.forEach(product => {
            const productEl = document.createElement('div');
            productEl.className = 'suggested-product';
            productEl.innerHTML = `
                <span class="product-name">${product.name}</span>
                <span class="product-price">${formatGs(product.price)} Gs.</span>
                <span class="suggestion-reason">${product.reason}</span>
            `;
            suggestionsContainer.appendChild(productEl);
        });
        
    } catch (error) {
        document.getElementById('suggested-products-list').innerHTML = '<p class="error">Error cargando sugerencias</p>';
    }
}

// Start photo capture
async function startPhotoCapture() {
    try {
        const cameraSection = document.getElementById('camera-section');
        const modalVideo = document.getElementById('modal-video');
        
        if (!cameraSection || !modalVideo) return;
        
        // Stop any existing stream first
        if (adminVideoStream) {
            adminVideoStream.getTracks().forEach(track => track.stop());
        }
        
        adminVideoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }
        });
        
        modalVideo.srcObject = adminVideoStream;
        cameraSection.classList.remove('hidden');
        
    } catch (error) {
        showNotification('Error accessing camera: ' + error.message, 'error');
    }
}

// Capture photo
async function capturePhoto() {
    try {
        const modalVideo = document.getElementById('modal-video');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = modalVideo.videoWidth;
        canvas.height = modalVideo.videoHeight;
        ctx.drawImage(modalVideo, 0, 0);
        
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
        
        // Update photo preview
        const photoUrl = URL.createObjectURL(blob);
        document.getElementById('current-student-photo').src = photoUrl;
        
        // Store blob for saving
        currentStudent.newPhoto = blob;
        
        cancelPhotoCapture();
        showNotification('Foto capturada. Recuerda guardar los cambios.', 'success');
        
    } catch (error) {
        showNotification('Error capturing photo: ' + error.message, 'error');
    }
}

// Cancel photo capture
function cancelPhotoCapture() {
    const cameraSection = document.getElementById('camera-section');
    cameraSection.classList.add('hidden');
    
    if (adminVideoStream) {
        adminVideoStream.getTracks().forEach(track => track.stop());
        adminVideoStream = null;
    }
}

// Add credits to student
async function addCredits() {
    const creditsInput = document.getElementById('add-credits');
    const credits = parseInt(creditsInput.value);
    
    if (!credits || credits <= 0) {
        showNotification('Ingresa una cantidad válida de créditos', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/students/${currentStudent.id}/add-credits`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ amount: credits })
        });
        
        if (!response.ok) throw new Error('Error adding credits');
        
        const result = await response.json();
        
        // Update balance display
        document.getElementById('edit-balance').value = result.new_balance;
        currentStudent.balance = result.new_balance;
        
        creditsInput.value = '';
        showNotification(`Agregados ${formatGs(credits)} Gs. Nuevo saldo: ${formatGs(result.new_balance)} Gs.`, 'success');
        
    } catch (error) {
        showNotification('Error adding credits: ' + error.message, 'error');
    }
}

// Save student changes
async function saveStudent() {
    try {
        const formData = new FormData();
        formData.append('name', document.getElementById('edit-name').value);
        formData.append('grade', document.getElementById('edit-grade').value);
        formData.append('balance', document.getElementById('edit-balance').value);
        
        if (currentStudent.newPhoto) {
            formData.append('photo', currentStudent.newPhoto, 'photo.jpg');
        }
        
        const response = await fetch(`/api/students/${currentStudent.id}`, {
            method: 'PUT',
            body: formData
        });
        
        if (!response.ok) throw new Error('Error saving student');
        
        showNotification('Estudiante actualizado correctamente', 'success');
        closeStudentModal();
        loadStudents(); // Refresh the list
        
    } catch (error) {
        showNotification('Error saving student: ' + error.message, 'error');
    }
}

// Delete student
async function deleteStudent() {
    showDeleteConfirmation();
}

// Show custom delete confirmation modal
function showDeleteConfirmation() {
    if (!currentStudent || !currentStudent.id) {
        showNotification('No hay alumno seleccionado para eliminar', 'error');
        return;
    }
    const modal = document.getElementById('custom-confirmation-modal');
    modal.innerHTML = `
        <div class="modal-content">
            <h2>¿Estás seguro de que quieres eliminar este estudiante?</h2>
            <p>Esta acción no se puede deshacer.</p>
            <button class="delete-btn" onclick="confirmDelete('${currentStudent.id}')">Eliminar</button>
            <button class="cancel-btn" onclick="closeDeleteConfirmation()">Cancelar</button>
        </div>
    `;
    modal.classList.remove('hidden');
}

async function confirmDelete(studentId) {
    try {
        if (!studentId) {
            throw new Error('No hay alumno seleccionado para eliminar');
        }

        const response = await fetch(`/api/students/${studentId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Error deleting student');
        
        showNotification('Estudiante eliminado correctamente', 'success');
        closeStudentModal();
        loadStudents(); // Refresh the list
        
    } catch (error) {
        showNotification('Error deleting student: ' + error.message, 'error');
    }
    closeDeleteConfirmation();
}

function closeDeleteConfirmation() {
    const modal = document.getElementById('custom-confirmation-modal');
    modal.classList.add('hidden');
}

// Close student modal
function closeStudentModal() {
    studentModal.classList.add('hidden');
    studentModal.style.display = '';
    cancelPhotoCapture();
    currentStudent = null;
}

// Show add product modal
function showAddProductModal() {
    const modal = document.getElementById('custom-confirmation-modal');
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>Agregar Nuevo Producto</h3>
                <button class="close-btn" onclick="closeAddProductModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="existing-products-notice">
                    <small style="color: #666; margin-bottom: 1rem; display: block;">
                        <strong>Nota:</strong> No puedes agregar productos con nombres duplicados.
                        Los productos existentes incluyen: Sandwich, Apple, Orange Juice, Yogurt, Cookie, Banana, Milk, Croissant, Water
                    </small>
                </div>
                <form id="add-product-form">
                    <div class="form-group">
                        <label for="product-name">Nombre del Producto:</label>
                        <input type="text" id="product-name" required placeholder="Ej: Pizza, Hamburguesa, etc." />
                    </div>
                    <div class="form-group">
                        <label for="product-price">Precio (Gs.):</label>
                        <input type="number" id="product-price" min="1" step="1" required placeholder="350" />
                    </div>
                    <div class="form-group">
                        <label for="product-stock">Stock inicial:</label>
                        <input type="number" id="product-stock" min="0" step="1" required placeholder="0" />
                    </div>
                    <div class="form-group">
                        <label for="product-min-stock">Alerta de stock mínimo:</label>
                        <input type="number" id="product-min-stock" min="0" step="1" required placeholder="20" value="20" />
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <button class="primary-btn" onclick="addProduct()">💾 Agregar Producto</button>
                <button class="secondary-btn" onclick="closeAddProductModal()">Cancelar</button>
            </div>
        </div>
    `;
    modal.classList.remove('hidden');
}

// Close add product modal
function closeAddProductModal() {
    const modal = document.getElementById('custom-confirmation-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

// Add new product
async function addProduct() {
    const name = document.getElementById('product-name').value.trim();
    const price = parseInt(document.getElementById('product-price').value);
    const stock = parseInt(document.getElementById('product-stock').value) || 0;
    const minStockInput = document.getElementById('product-min-stock');
    const minStock = parseInt(minStockInput.value);

    if (!name || !price || price <= 0) {
        showNotification('Por favor ingresa un nombre y precio válido', 'error');
        return;
    }

    if (isNaN(minStock) || minStock < 0) {
        showNotification('El stock mínimo debe ser 0 o mayor', 'error');
        return;
    }

    try {
        const response = await fetch('/api/products', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                price: price,
                stock: stock,
                default_min_stock: minStock
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || `Error ${response.status}: ${response.statusText}`;
            throw new Error(errorMessage);
        }

        const result = await response.json();
        console.log('Product created:', result);
        showNotification('Producto agregado correctamente', 'success');
        closeAddProductModal();

        // Clear form
        document.getElementById('product-name').value = '';
        document.getElementById('product-price').value = '';
        document.getElementById('product-stock').value = '';
        minStockInput.value = '20';

        // Refresh the products list with a small delay
        setTimeout(() => {
            loadProducts();
        }, 100);

    } catch (error) {
        console.error('Error adding product:', error);
        showNotification(error.message, 'error');
    }
}

// Load products (placeholder)
async function loadProducts() {
    try {
        const response = await fetchWithAuth('/api/products');
        const products = await response.json();
        currentProducts = Array.isArray(products) ? products : [];
        renderProducts(currentProducts);

        if (productSearchInput && productSearchInput.value.trim()) {
            filterProducts(productSearchInput.value);
        }

    } catch (error) {
        showNotification('Error loading products: ' + error.message, 'error');
    }
}

function renderProducts(products) {
    const productsList = document.getElementById('products-list');
    if (!productsList) return;

    productsList.innerHTML = '';

    if (!products || products.length === 0) {
        productsList.innerHTML = '<p class="no-data">No hay productos disponibles</p>';
        return;
    }

    products.forEach((product) => {
        const safeName = (product.name || '').replace(/'/g, "\\'");
        const productEl = document.createElement('div');
        productEl.className = 'product-row';
        productEl.innerHTML = `
            <div class="product-name">${product.name}</div>
            <div class="product-price">${formatGs(product.price)} Gs.</div>
            <div class="product-stock">Stock: ${product.stock ?? 0}</div>
            <div class="product-actions">
                <button class="edit-btn" onclick="editProduct(${product.id}, '${safeName}', ${product.price}, ${product.stock ?? 0}, ${product.default_min_stock ?? 0})">✏️ Editar</button>
            </div>
        `;
        productsList.appendChild(productEl);
    });
}

function filterProducts(query) {
    clearTimeout(productSearchDebounceTimer);
    productSearchDebounceTimer = setTimeout(() => {
        const trimmed = (query || '').trim().toLowerCase();
        if (!trimmed) {
            renderProducts(currentProducts);
            return;
        }

        const filtered = currentProducts.filter((product) => {
            const name = (product.name || '').toLowerCase();
            const price = String(product.price || '').toLowerCase();
            return name.includes(trimmed) || price.includes(trimmed);
        });

        renderProducts(filtered);
    }, DEBOUNCE_DELAY);
}

// Edit product
function editProduct(id, name, price, stock, minStock) {
    const modal = document.getElementById('custom-confirmation-modal');
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>Editar Producto</h3>
                <button class="close-btn" onclick="closeAddProductModal()">&times;</button>
            </div>
            <div class="modal-body">
                <form id="edit-product-form">
                    <div class="form-group">
                        <label for="edit-product-name">Nombre del Producto:</label>
                        <input type="text" id="edit-product-name" value="${name}" required />
                    </div>
                    <div class="form-group">
                        <label for="edit-product-price">Precio (Gs.):</label>
                        <input type="number" id="edit-product-price" value="${price}" min="1" step="1" required />
                    </div>
                    <div class="form-group">
                        <label for="edit-product-stock">Stock:</label>
                        <input type="number" id="edit-product-stock" value="${stock}" min="0" step="1" required />
                    </div>
                    <div class="form-group">
                        <label for="edit-product-min-stock">Alerta de stock mínimo:</label>
                        <input type="number" id="edit-product-min-stock" value="${minStock}" min="0" step="1" required />
                    </div>
                    <input type="hidden" id="edit-product-id" value="${id}" />
                </form>
            </div>
            <div class="modal-footer">
                <button class="primary-btn" onclick="updateProduct()"> Guardar Cambios</button>
                <button class="secondary-btn" onclick="closeAddProductModal()">Cancelar</button>
            </div>
        </div>
    `;
    modal.classList.remove('hidden');
}

// Update product
async function updateProduct() {
    const id = document.getElementById('edit-product-id').value;
    const name = document.getElementById('edit-product-name').value.trim();
    const price = parseInt(document.getElementById('edit-product-price').value);
    const stock = parseInt(document.getElementById('edit-product-stock').value);
    const minStock = parseInt(document.getElementById('edit-product-min-stock').value);

    if (!name || !price || price <= 0 || isNaN(stock) || stock < 0) {
        showNotification('Por favor ingresa un nombre, precio y stock válido', 'error');
        return;
    }

    if (isNaN(minStock) || minStock < 0) {
        showNotification('El stock mínimo debe ser 0 o mayor', 'error');
        return;
    }

    try {
        const response = await fetchWithAuth(`/api/products/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                price: price,
                stock: stock,
                default_min_stock: minStock,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || `Error ${response.status}: ${response.statusText}`;
            throw new Error(errorMessage);
        }

        showNotification('Producto actualizado correctamente', 'success');
        closeAddProductModal();
        loadProducts();
    } catch (error) {
        showNotification('Error actualizando producto: ' + error.message, 'error');
    }
}

// Delete product
function deleteProduct(id, name) {
    const modal = document.getElementById('custom-confirmation-modal');
    modal.innerHTML = `
        <div class="modal-content">
            <h2>¿Eliminar producto?</h2>
            <p>¿Estás seguro de que quieres eliminar "<strong>${name}</strong>"?</p>
            <p><small>Esta acción requiere implementación en el backend.</small></p>
            <button class="delete-btn" onclick="confirmDeleteProduct(${id}, '${name}')">Eliminar</button>
            <button class="cancel-btn" onclick="closeAddProductModal()">Cancelar</button>
        </div>
    `;
    modal.classList.remove('hidden');
}

// Confirm delete product
async function confirmDeleteProduct(id, name) {
    try {
        // Note: This would require a backend endpoint like DELETE /api/products/{id}
        // For now, we'll show the concept
        showNotification(`Eliminación de "${name}" requiere endpoint DELETE /api/products/${id}`, 'info');
        closeAddProductModal();
        loadProducts();

    } catch (error) {
        showNotification('Error eliminando producto: ' + error.message, 'error');
    }
}

async function loadTransactions() {
    const list = document.getElementById('transactions-list');
    if (!list) return;

    const filterSelect = document.getElementById('transaction-filter');
    const dateFromInput = document.getElementById('date-from');
    const dateToInput = document.getElementById('date-to');

    const params = new URLSearchParams();
    const studentId = filterSelect ? filterSelect.value : '';
    const dateFrom = dateFromInput && dateFromInput.value ? dateFromInput.value : '';
    const dateTo = dateToInput && dateToInput.value ? dateToInput.value : '';

    if (studentId) params.append('student_id', studentId);
    if (dateFrom) params.append('date_from', dateFrom);
    if (dateTo) params.append('date_to', dateTo);

    let url = '/api/transactions';
    const qs = params.toString();
    if (qs) url += `?${qs}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            list.innerHTML = '<p class="error">Error cargando transacciones</p>';
            return;
        }

        const transactions = await response.json();
        if (!transactions || transactions.length === 0) {
            list.innerHTML = '<p class="no-data">Sin transacciones para el criterio seleccionado</p>';
            return;
        }

        list.innerHTML = '';
        const fragment = document.createDocumentFragment();

        transactions.forEach(t => {
            const row = document.createElement('div');
            row.className = 'transaction-item';
            row.innerHTML = `
                <div class="transaction-info">
                    <span class="product-name">${t.product_name}</span>
                    <span class="transaction-date">${new Date(t.created_at).toLocaleString()}</span>
                    <span class="transaction-student">${t.student_name}</span>
                </div>
                <div class="transaction-amount">-${formatGs(t.amount)} Gs.</div>
            `;
            fragment.appendChild(row);
        });

        list.appendChild(fragment);
    } catch (error) {
        list.innerHTML = '<p class="error">Error cargando transacciones</p>';
    }
}

// Load analytics (placeholder)
async function loadAnalytics() {
    const summaryNodes = {
        totalSales: document.getElementById('summary-total-sales'),
        totalTransactions: document.getElementById('summary-total-transactions'),
        averageTicket: document.getElementById('summary-average-ticket'),
        uniqueStudents: document.getElementById('summary-unique-students'),
        bestDayLabel: document.getElementById('summary-best-day-label'),
        bestDayAmount: document.getElementById('summary-best-day-amount'),
    };

    const miniLists = {
        products: document.getElementById('top-products-list'),
        students: document.getElementById('top-students-list'),
    };

    const emptyStates = {
        products: document.getElementById('top-products-empty'),
        students: document.getElementById('top-students-empty'),
        daily: document.getElementById('daily-sales-empty'),
    };

    const chartCanvases = {
        products: document.getElementById('top-products-chart'),
        students: document.getElementById('top-students-chart'),
        daily: document.getElementById('daily-sales-chart'),
    };

    if (!summaryNodes.totalSales || !miniLists.products || !chartCanvases.daily) {
        return;
    }

    try {
        const response = await fetch('/api/analytics/summary');
        if (!response.ok) {
            Object.values(emptyStates).forEach(node => {
                if (node) {
                    node.textContent = 'Error cargando analíticas';
                    node.classList.remove('hidden');
                }
            });
            return;
        }

        const data = await response.json();
        const topProductsData = data.top_products || [];
        const topStudentsData = data.top_students || [];
        const dailySalesData = data.daily_sales || [];
        const summary = data.summary || {};

        renderSummaryCards(summaryNodes, summary);
        renderMiniList(miniLists.products, topProductsData, 'ventas');
        renderMiniList(miniLists.students, topStudentsData, 'compras', true);
        toggleEmptyState(emptyStates.products, topProductsData.length === 0);
        toggleEmptyState(emptyStates.students, topStudentsData.length === 0);
        toggleEmptyState(emptyStates.daily, dailySalesData.length === 0);

        document.getElementById('top-products-count').textContent = topProductsData.length;
        document.getElementById('top-students-count').textContent = topStudentsData.length;

        renderChart('products', chartCanvases.products, topProductsData.map(p => p.name), topProductsData.map(p => p.total_amount));
        renderChart('students', chartCanvases.students, topStudentsData.map(s => s.name), topStudentsData.map(s => s.total_spent));
        renderDailySalesChart(chartCanvases.daily, dailySalesData);
    } catch (error) {
        Object.values(emptyStates).forEach(node => {
            if (node) {
                node.textContent = 'Error cargando analíticas';
                node.classList.remove('hidden');
            }
        });
        console.error('loadAnalytics error:', error);
    }
}

function renderSummaryCards(nodes, summary) {
    if (!nodes) return;

    const totalSales = summary.total_sales ?? 0;
    const totalTransactions = summary.total_transactions ?? 0;
    const averageTicket = summary.average_ticket ?? 0;
    const uniqueStudents = summary.unique_students ?? 0;
    const bestDay = summary.best_day;

    if (nodes.totalSales) nodes.totalSales.textContent = `${formatGs(totalSales)} Gs.`;
    if (nodes.totalTransactions) nodes.totalTransactions.textContent = totalTransactions;
    if (nodes.averageTicket) nodes.averageTicket.textContent = `${formatGs(Math.round(averageTicket))} Gs.`;
    if (nodes.uniqueStudents) nodes.uniqueStudents.textContent = uniqueStudents;

    if (nodes.bestDayLabel && nodes.bestDayAmount) {
        if (bestDay && bestDay.date) {
            const label = new Date(bestDay.date).toLocaleDateString('es-PY', {
                weekday: 'short', day: '2-digit', month: 'short'
            });
            nodes.bestDayLabel.textContent = label;
            nodes.bestDayAmount.textContent = `${formatGs(bestDay.total_amount)} Gs. vendidos`;
        } else {
            nodes.bestDayLabel.textContent = '--';
            nodes.bestDayAmount.textContent = 'Sin ventas registradas';
        }
    }
}

function renderMiniList(container, data, label, isStudent = false) {
    if (!container) return;
    container.innerHTML = '';

    if (!data || data.length === 0) {
        container.innerHTML = '<li class="muted">Sin datos disponibles</li>';
        return;
    }

    data.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `
            <div>
                <strong>${item.name}</strong>
                <span class="muted">${item.transaction_count} ${label}</span>
            </div>
            <span class="value">${formatGs(isStudent ? item.total_spent : item.total_amount)} Gs.</span>
        `;
        container.appendChild(li);
    });
}

function toggleEmptyState(node, isEmpty) {
    if (!node) return;
    if (isEmpty) {
        node.classList.remove('hidden');
    } else {
        node.classList.add('hidden');
    }
}

function renderChart(type, canvas, labels, values) {
    if (!canvas || typeof Chart === 'undefined') return;
    const chartConfig = {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Guaraníes',
                    data: values,
                    backgroundColor: 'rgba(5, 150, 105, 0.5)',
                    borderColor: 'rgba(5, 150, 105, 1)',
                    borderWidth: 2,
                    borderRadius: 8,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    ticks: {
                        callback: (val) => formatCompactNumber(val),
                        color: '#cbd5e1'
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.2)' }
                },
                x: {
                    ticks: { color: '#cbd5e1' },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: context => `${context.dataset.label}: ${formatGs(context.parsed.y ?? context.parsed)} Gs.`
                    }
                }
            }
        }
    };

    if (type === 'products' && topProductsChartInstance) topProductsChartInstance.destroy();
    if (type === 'students' && topStudentsChartInstance) topStudentsChartInstance.destroy();

    if (type === 'products') {
        topProductsChartInstance = new Chart(canvas, chartConfig);
    } else if (type === 'students') {
        topStudentsChartInstance = new Chart(canvas, chartConfig);
    }
}

function renderDailySalesChart(canvas, data) {
    if (!canvas || typeof Chart === 'undefined') return;

    const labels = data.map(item => item.date);
    const salesValues = data.map(item => item.total_amount);

    const config = {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Ventas (Gs.)',
                    data: salesValues,
                    borderColor: 'rgba(59, 130, 246, 1)',
                    backgroundColor: 'rgba(59, 130, 246, 0.25)',
                    tension: 0.35,
                    fill: true,
                    borderWidth: 3,
                    pointRadius: 4,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { callback: (val) => formatCompactNumber(val), color: '#cbd5e1' },
                    grid: { color: 'rgba(148, 163, 184, 0.2)' }
                },
                x: {
                    ticks: { color: '#cbd5e1' },
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { labels: { color: '#e2e8f0' } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.datasetIndex === 0) {
                                return `${ctx.dataset.label}: ${formatGs(ctx.parsed.y)} Gs.`;
                            }
                            return `${ctx.dataset.label}: ${ctx.parsed.y}`;
                        }
                    }
                }
            }
        }
    };

    if (dailySalesChartInstance) dailySalesChartInstance.destroy();
    dailySalesChartInstance = new Chart(canvas, config);
}

function formatCompactNumber(value) {
    if (!value && value !== 0) return '';
    if (Math.abs(value) >= 1_000_000) {
        return `${(value / 1_000_000).toFixed(1)}M`;
    }
    if (Math.abs(value) >= 1_000) {
        return `${(value / 1_000).toFixed(1)}K`;
    }
    return String(value);
}

// View student history
async function viewHistory(studentId) {
    try {
        await editStudent(studentId);
        const modalTitle = document.getElementById('modal-title');
        if (modalTitle) modalTitle.textContent = 'Historial de Compras';
        const purchasesContainer = document.getElementById('student-purchases');
        if (purchasesContainer) {
            // Give a small delay to allow async content to render
            setTimeout(() => {
                purchasesContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
                purchasesContainer.classList.add('pulse-highlight');
                setTimeout(() => purchasesContainer.classList.remove('pulse-highlight'), 1200);
            }, 150);
        }
    } catch (error) {
        showNotification('Error mostrando historial: ' + error.message, 'error');
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

// Setup navigation
function setupNavigation() {
    const studentsBtn = document.querySelector('[data-section="students"]');
    const studentsSection = document.getElementById('students-section');
    if (studentsBtn) studentsBtn.classList.add('active');
    if (studentsSection) studentsSection.classList.add('active');
    currentSection = null;
    switchSection('students');
}
