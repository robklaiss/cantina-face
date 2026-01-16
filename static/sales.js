// Sales view script
let products = [];
let cart = [];
let currentStudent = null;
let productSearchTimeout = null;

const productButtons = document.querySelectorAll('.product-btn');
const productsGrid = document.getElementById('products-grid');
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const cartItems = document.getElementById('cart-items');
const cartTotal = document.getElementById('cart-total');
const manualCharge = document.getElementById('manual-charge');
const chargeBtn = document.getElementById('charge-btn');
const saleClearBtn = document.getElementById('sale-clear-btn');
const studentCard = document.getElementById('student-card');
const studentPhoto = document.getElementById('student-photo');
const studentName = document.getElementById('student-name');
const studentGrade = document.getElementById('student-grade');
const studentBalance = document.getElementById('student-balance');
const noStudent = document.getElementById('no-student');
const activeStudentName = document.getElementById('active-student-name');
const notifications = document.getElementById('notifications');
const urlParams = new URLSearchParams(window.location.search);
const initialStudentId = urlParams.get('student_id');

function formatGs(amount) {
    const n = Number(amount) || 0;
    try {
        return n.toLocaleString('es-PY', { maximumFractionDigits: 0 });
    } catch (_) {
        return n.toString();
    }
}

function showNotification(message, type = 'info') {
    if (!notifications) return;
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notifications.appendChild(notification);
    setTimeout(() => notification.remove(), 4000);
}

async function loadProducts() {
    try {
        const response = await fetch('/api/products');
        products = await response.json();
        productButtons.forEach((btn, index) => {
            const product = products[index];
            if (product) {
                btn.dataset.productId = String(product.id);
                btn.innerHTML = `
                    <span class="kbd-index">${index + 1}</span>
                    <span class="label">${product.name}</span>
                    <span class="price">${formatGs(product.price)} Gs.</span>
                `;
                btn.title = `${product.name} — ${formatGs(product.price)} Gs.`;
                btn.disabled = false;
                btn.classList.remove('disabled');
            } else {
                btn.removeAttribute('data-product-id');
                btn.innerHTML = `<span class="label">—</span>`;
                btn.title = 'Sin asignar';
                btn.disabled = true;
                btn.classList.add('disabled');
            }
        });
    } catch (error) {
        showNotification('No se pudieron cargar productos', 'error');
    }
}

function renderCart() {
    if (!cartItems || !cartTotal) return;
    cartItems.innerHTML = '';
    if (cart.length === 0) {
        cartItems.innerHTML = '<p class="no-data">Sin productos seleccionados</p>';
        cartTotal.textContent = '0';
        manualCharge?.classList.add('hidden');
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
        row.querySelector('.qty-btn.minus')?.addEventListener('click', () => updateCartQuantity(item.productId, -1));
        row.querySelector('.qty-btn.plus')?.addEventListener('click', () => updateCartQuantity(item.productId, 1));
        cartItems.appendChild(row);
    });
    cartTotal.textContent = formatGs(total);
}

function updateCartQuantity(productId, delta) {
    const entry = cart.find(item => item.productId === productId);
    if (!entry) return;
    entry.quantity += delta;
    if (entry.quantity <= 0) {
        cart = cart.filter(item => item.productId !== productId);
    }
    renderCart();
}

function addToCart(productId) {
    if (!currentStudent) {
        showNotification('Seleccioná un alumno antes de cargar productos', 'warning');
        return;
    }
    const existing = cart.find(item => item.productId === productId);
    if (existing) {
        existing.quantity += 1;
    } else {
        cart.push({ productId, quantity: 1 });
    }
    renderCart();
    manualCharge?.classList.remove('hidden');
}

function clearSaleState() {
    currentStudent = null;
    cart = [];
    renderCart();
    manualCharge?.classList.add('hidden');
    studentCard?.classList.add('hidden');
    noStudent?.classList.remove('hidden');
    activeStudentName && (activeStudentName.textContent = 'Ninguna');
    updateStudentQueryParam(null);
}

function showStudentCard(student) {
    currentStudent = student;
    if (studentPhoto) studentPhoto.src = student.photo_url || '/default-avatar.png';
    if (studentName) studentName.textContent = student.name;
    if (studentGrade) studentGrade.textContent = student.grade || '-';
    if (studentBalance) studentBalance.textContent = formatGs(student.balance);
    studentCard?.classList.remove('hidden');
    noStudent?.classList.add('hidden');
    manualCharge?.classList.remove('hidden');
    if (activeStudentName) activeStudentName.textContent = student.name;
    updateStudentQueryParam(student.id);
}

function updateStudentQueryParam(studentId) {
    const url = new URL(window.location.href);
    if (studentId) {
        url.searchParams.set('student_id', studentId);
    } else {
        url.searchParams.delete('student_id');
    }
    window.history.replaceState({}, '', url);
}

async function loadStudentById(studentId) {
    if (!studentId) return;
    try {
        const response = await fetch(`/api/students/${studentId}`);
        if (!response.ok) {
            throw new Error('Alumno no encontrado');
        }
        const student = await response.json();
        showStudentCard(student);
    } catch (error) {
        showNotification(error.message || 'No se pudo cargar el alumno', 'error');
    }
}

function displayProductSearchResults(list) {
    if (!searchResults) return;
    searchResults.innerHTML = '';
    if (list.length === 0) {
        searchResults.innerHTML = '<p class="no-data">Sin productos coincidentes</p>';
        return;
    }
    list.forEach(product => {
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
            addToCart(product.id);
            searchInput.value = '';
            searchResults.innerHTML = '';
        });
        searchResults.appendChild(div);
    });
}

function searchProducts(query) {
    const trimmed = (query || '').trim().toLowerCase();
    if (!trimmed) {
        searchResults.innerHTML = '';
        return;
    }
    if (!products.length) {
        searchResults.innerHTML = '<p class="no-data">Sin productos cargados</p>';
        return;
    }
    const matches = products.filter(product => {
        const name = (product.name || '').toLowerCase();
        const priceStr = String(product.price || '');
        return name.includes(trimmed) || priceStr.includes(trimmed);
    });
    displayProductSearchResults(matches);
}

async function chargeCart() {
    if (!currentStudent) {
        showNotification('Seleccioná un alumno', 'warning');
        return;
    }
    if (cart.length === 0) {
        showNotification('No hay productos para cobrar', 'warning');
        return;
    }
    let total = 0;
    for (const item of cart) {
        const product = products.find(p => p.id === item.productId);
        if (!product) continue;
        total += product.price * item.quantity;
    }
    if (currentStudent.balance < total) {
        showNotification('Saldo insuficiente', 'error');
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
                    throw new Error(result.detail || 'Error en el cobro');
                }
                if (typeof result.new_balance === 'number') {
                    currentStudent.balance = result.new_balance;
                    studentBalance.textContent = formatGs(currentStudent.balance);
                }
            }
        }
        cart = [];
        renderCart();
        showNotification(`Cobro realizado a ${currentStudent.name}`, 'success');
    } catch (error) {
        showNotification(error.message || 'Error realizando el cobro', 'error');
    }
}

function setupEventListeners() {
    productButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const productId = parseInt(btn.dataset.productId, 10);
            if (!productId) return;
            addToCart(productId);
        });
    });

    if (searchInput) {
        searchInput.addEventListener('input', (event) => {
            clearTimeout(productSearchTimeout);
            const value = event.target.value;
            productSearchTimeout = setTimeout(() => searchProducts(value), 200);
        });
    }

    chargeBtn?.addEventListener('click', chargeCart);
    saleClearBtn?.addEventListener('click', clearSaleState);
}

async function initSalesView() {
    setupEventListeners();
    await loadProducts();
    renderCart();
    if (initialStudentId) {
        await loadStudentById(initialStudentId);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSalesView);
} else {
    initSalesView();
}
