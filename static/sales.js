// Sales view script
let products = [];
let cart = [];
let currentStudent = null;
let productSearchTimeout = null;
let currentCartTotal = 0;
let dispatchItems = [];
let dispatchActive = false;
let scheduledOrders = [];
let dispatchContext = null;

const AUTH_TOKEN_KEY = 'cantina_face_token';
const AUTH_USER_KEY = 'cantina_face_user';
const AUTH_PROFILE_KEY = 'cantina_face_user_profile';
let salesAuthToken = localStorage.getItem(AUTH_TOKEN_KEY) || null;
let salesAuthEmail = localStorage.getItem(AUTH_USER_KEY) || null;
let salesAuthProfile = loadProfileFromStorage();
const nativeFetch = window.fetch.bind(window);

const productButtons = document.querySelectorAll('.product-btn');
const productsGrid = document.getElementById('products-grid');
const searchZone = document.getElementById('search-zone');
const productsZone = document.getElementById('products-zone');
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const cartItems = document.getElementById('cart-items');
const cartTotal = document.getElementById('cart-total');
const manualCharge = document.getElementById('manual-charge');
const chargeBtn = document.getElementById('charge-btn');
const manualChargeBtn = document.getElementById('manual-charge-btn');
const saleClearBtn = document.getElementById('sale-clear-btn');
const dispatchPanel = document.getElementById('dispatch-panel');
const dispatchList = document.getElementById('dispatch-list');
const dispatchClearBtn = document.getElementById('dispatch-clear-btn');
const dispatchAddBtn = document.getElementById('dispatch-add-btn');
const scheduledPanel = document.getElementById('scheduled-orders-panel');
const scheduledList = document.getElementById('scheduled-orders-list');
const manualChargeModal = document.getElementById('manual-charge-modal');
const manualTotalLabel = document.getElementById('manual-total');
const manualCashInput = document.getElementById('manual-cash-input');
const manualChangeOutput = document.getElementById('manual-change-output');
const manualChargeConfirm = document.getElementById('manual-charge-confirm');
const manualChargeCancel = document.getElementById('manual-charge-cancel');
const cashOptionsContainer = document.getElementById('cash-options');
const manualExactBtn = document.getElementById('manual-exact-btn');
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

function currentPathWithFallback() {
    const path = `${window.location.pathname}${window.location.search}`;
    if (!path || path === '/' || path.startsWith('/login')) {
        return '/sales';
    }
    return path;
}

// Scheduled orders helpers
async function loadScheduledOrdersForStudent(studentId) {
    if (!studentId || !scheduledList) return;
    try {
        scheduledList.innerHTML = '<p class="muted">Cargando pedidos...</p>';
        const response = await fetchWithAuth(`/api/students/${studentId}/scheduled-orders?status_filter=pending`);
        if (!response.ok) {
            throw new Error('No se pudo cargar pedidos programados');
        }
        scheduledOrders = await response.json();
        renderScheduledOrders();
    } catch (error) {
        scheduledOrders = [];
        renderScheduledOrders(error.message || 'No se pudo cargar pedidos');
    }
}

function renderScheduledOrders(errorMessage = null) {
    if (!scheduledPanel || !scheduledList) return;
    if (!currentStudent) {
        scheduledPanel.classList.add('hidden');
        scheduledList.innerHTML = '<p class="muted">Seleccioná un alumno para ver pedidos.</p>';
        return;
    }
    scheduledPanel.classList.remove('hidden');
    if (errorMessage) {
        scheduledList.innerHTML = `<p class="error">${errorMessage}</p>`;
        return;
    }
    if (!scheduledOrders.length) {
        scheduledList.innerHTML = '<p class="muted">Sin pedidos pendientes.</p>';
        return;
    }

    scheduledList.innerHTML = '';
    scheduledOrders.forEach((order) => {
        const card = document.createElement('article');
        card.className = 'scheduled-card';
        card.innerHTML = `
            <header>
                <div>
                    <strong>${formatDate(order.scheduled_for)}</strong>
                    <span>${order.items.map((item) => `${item.quantity}x ${item.product_name || 'Producto'}`).join(', ')}</span>
                </div>
                <button class="primary-btn" data-dispatch="${order.id}">Despachar</button>
            </header>
            ${order.notes ? `<p class="note">Mensaje: ${order.notes}</p>` : ''}
        `;
        scheduledList.appendChild(card);
    });

    scheduledList.querySelectorAll('button[data-dispatch]').forEach((btn) => {
        btn.addEventListener('click', () => openScheduledDispatch(btn.dataset.dispatch));
    });
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('es-PY', { weekday: 'short', day: 'numeric', month: 'short' });
}

async function dispatchScheduledOrder(orderId) {
    if (!orderId) return;
    try {
        const response = await fetchWithAuth(`/api/scheduled-orders/${orderId}/dispatch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'No se pudo despachar el pedido');
        }
        showNotification('Pedido despachado', 'success');
        scheduledOrders = scheduledOrders.filter((order) => String(order.id) !== String(orderId));
        renderScheduledOrders();
        return result;
    } catch (error) {
        showNotification(error.message || 'Error despachando pedido', 'error');
        throw error;
    }
}

function openScheduledDispatch(orderId) {
    if (!orderId) return;
    const order = scheduledOrders.find((o) => String(o.id) === String(orderId));
    if (!order) {
        showNotification('Pedido no encontrado', 'error');
        return;
    }
    const items = (order.items || []).map((item) => ({
        productId: item.product_id,
        quantity: item.quantity,
        label: item.product_name || 'Producto',
    }));
    if (!items.length) {
        showNotification('Este pedido no tiene productos para despachar', 'warning');
        return;
    }
    showDispatchPanel(items, { type: 'scheduled', orderId: order.id });
}

function redirectToLogin() {
    const next = encodeURIComponent(currentPathWithFallback());
    window.location.href = `/login.html?next=${next}`;
}

function loadProfileFromStorage() {
    try {
        const raw = localStorage.getItem(AUTH_PROFILE_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (error) {
        console.warn('[sales] No se pudo parsear el perfil guardado', error);
        return null;
    }
}

function setSalesProfile(profile) {
    salesAuthProfile = profile || null;
    if (profile) {
        localStorage.setItem(AUTH_PROFILE_KEY, JSON.stringify(profile));
    } else {
        localStorage.removeItem(AUTH_PROFILE_KEY);
    }
}

function setSalesSession(token, email) {
    salesAuthToken = token || null;
    salesAuthEmail = email || null;
    if (salesAuthToken) {
        localStorage.setItem(AUTH_TOKEN_KEY, salesAuthToken);
        if (salesAuthEmail) {
            localStorage.setItem(AUTH_USER_KEY, salesAuthEmail);
        }
    } else {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
        setSalesProfile(null);
    }
}

function handleUnauthorized() {
    showNotification('Sesión expirada, vuelve a ingresar', 'error');
    setSalesSession(null, null);
    redirectToLogin();
}

async function fetchWithAuth(url, options = {}) {
    const headers = options.headers ? { ...options.headers } : {};
    if (salesAuthToken && !headers['Authorization']) {
        headers['Authorization'] = `Bearer ${salesAuthToken}`;
    }
    const response = await nativeFetch(url, { ...options, headers });
    if (response.status === 401) {
        handleUnauthorized();
    }
    return response;
}

function ensureAuthenticated() {
    if (!salesAuthToken) {
        redirectToLogin();
        return false;
    }
    return true;
}

async function ensureSalesProfile() {
    if (!salesAuthToken || salesAuthProfile) return;
    try {
        const response = await fetchWithAuth('/auth/me');
        if (response.ok) {
            const profile = await response.json();
            setSalesProfile(profile);
        }
    } catch (error) {
        console.error('[sales] No se pudo cargar el perfil actual:', error);
    }
}

function getActivePointOfSaleId() {
    const posId = salesAuthProfile?.point_of_sale_id;
    if (typeof posId === 'number' && !Number.isNaN(posId)) {
        return posId;
    }
    return 1; // Fallback al POS01
}

function formatGs(amount) {
    const n = Number(amount) || 0;
    try {
        return n.toLocaleString('es-PY', { maximumFractionDigits: 0 });
    } catch (_) {
        return n.toString();
    }
}

function renderCashOptions() {
    if (!cashOptionsContainer) return;
    const denominations = [100000, 50000, 10000, 5000, 1000];
    cashOptionsContainer.innerHTML = '';

    denominations
        .filter(value => value >= currentCartTotal)
        .forEach(value => {
            const btn = document.createElement('button');
            btn.className = 'secondary-btn';
            btn.type = 'button';
            btn.textContent = formatGs(value);
            btn.addEventListener('click', () => {
                manualCashInput.value = value;
                handleManualCashInput();
            });
            cashOptionsContainer.appendChild(btn);
        });
}

function openManualChargeModal() {
    if (!currentStudent) {
        showNotification('Seleccioná un alumno', 'warning');
        return;
    }
    if (!cart.length) {
        showNotification('No hay productos para cobrar', 'warning');
        return;
    }
    manualTotalLabel.textContent = `${formatGs(currentCartTotal)} Gs.`;
    manualCashInput.value = '';
    manualChangeOutput.textContent = '0 Gs.';
    renderCashOptions();
    manualChargeModal?.classList.remove('hidden');
    manualCashInput?.focus();
}

function closeManualChargeModal() {
    manualChargeModal?.classList.add('hidden');
    manualCashInput.value = '';
    manualChangeOutput.textContent = '0 Gs.';
}

function handleManualCashInput() {
    const cashValue = parseInt(manualCashInput.value, 10);
    const change = isNaN(cashValue) ? 0 : Math.max(0, cashValue - currentCartTotal);
    manualChangeOutput.textContent = `${formatGs(change)} Gs.`;
}

function confirmManualCharge() {
    if (!currentStudent) {
        showNotification('Seleccioná un alumno', 'warning');
        return;
    }
    if (!cart.length) {
        showNotification('No hay productos para cobrar', 'warning');
        return;
    }
    const cashValue = parseInt(manualCashInput.value, 10);
    if (isNaN(cashValue) || cashValue <= 0) {
        showNotification('Ingresá un monto en efectivo válido', 'error');
        return;
    }
    if (cashValue < currentCartTotal) {
        showNotification('El efectivo no cubre el total a cobrar', 'error');
        return;
    }
    const change = cashValue - currentCartTotal;
    const fulfilledItems = cart.map(item => ({ ...item }));

    (async () => {
        try {
            for (const item of cart) {
                const product = products.find(p => p.id === item.productId);
                if (!product) continue;
                for (let i = 0; i < item.quantity; i++) {
                    const response = await fetchWithAuth('/api/charge', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            student_id: currentStudent.id,
                            product_id: product.id,
                            payment_method: 'cash',
                            point_of_sale_id: getActivePointOfSaleId(),
                        })
                    });
                    const result = await response.json();
                    if (!response.ok) {
                        throw new Error(result.detail || 'Error registrando el cobro manual');
                    }
                }
            }
            showNotification(`Cobro manual registrado. Vuelto: ${formatGs(change)} Gs.`, 'success');
            cart = [];
            renderCart();
            closeManualChargeModal();
            showDispatchPanel(fulfilledItems);
        } catch (error) {
            showNotification(error.message || 'Error registrando el cobro manual', 'error');
        }
    })();
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
        const response = await fetchWithAuth('/api/products');
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
        currentCartTotal = 0;
        cartTotal.textContent = '0';
        manualCharge?.classList.add('hidden');
        if (!dispatchActive) {
            hideDispatchPanel();
        }
        renderDispatchChecklist();
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
    currentCartTotal = total;
    cartTotal.textContent = formatGs(total);
    renderDispatchChecklist();
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

function renderDispatchChecklist() {
    if (!dispatchList) return;
    dispatchList.innerHTML = '';

    if (!dispatchActive || !dispatchItems.length) {
        dispatchList.innerHTML = '<p class="muted">Sin productos cargados</p>';
        return;
    }

    dispatchItems.forEach(item => {
        const product = products.find(p => p.id === item.productId);
        const productLabel = product?.name || item.label || 'Producto';
        const wrapper = document.createElement('div');
        wrapper.className = 'dispatch-item';
        const title = document.createElement('div');
        title.className = 'dispatch-item-title';
        title.textContent = `${productLabel} (${item.quantity})`;
        wrapper.appendChild(title);

        const checklist = document.createElement('div');
        checklist.className = 'dispatch-checklist';
        for (let i = 0; i < item.quantity; i++) {
            const row = document.createElement('label');
            row.className = 'dispatch-check';

            const text = document.createElement('span');
            text.textContent = item.quantity > 1 ? `Unidad ${i + 1}` : 'Entregado';
            row.appendChild(text);

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            row.appendChild(checkbox);

            checklist.appendChild(row);
        }
        wrapper.appendChild(checklist);
        dispatchList.appendChild(wrapper);
    });
}

function showDispatchPanel(items, context = { type: 'manual' }) {
    function clearSaleState() {
        dispatchItems = [];
        dispatchContext = null;
        dispatchActive = false;
        dispatchPanel?.classList.add('hidden');
        searchZone?.classList.remove('hidden');
        productsZone?.classList.remove('hidden');
        if (dispatchAddBtn) dispatchAddBtn.classList.remove('hidden');
        renderDispatchChecklist();
    }

    dispatchItems = items;
    dispatchContext = context;
    dispatchActive = true;
    dispatchPanel?.classList.remove('hidden');
    searchZone?.classList.add('hidden');
    productsZone?.classList.add('hidden');
    if (dispatchAddBtn) dispatchAddBtn.classList.toggle('hidden', context.type === 'scheduled');
    renderDispatchChecklist();
}

function hideDispatchPanel() {
    dispatchItems = [];
    dispatchContext = null;
    dispatchActive = false;
    dispatchPanel?.classList.add('hidden');
    searchZone?.classList.remove('hidden');
    productsZone?.classList.remove('hidden');
    if (dispatchAddBtn) dispatchAddBtn.classList.remove('hidden');
    renderDispatchChecklist();
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
    scheduledOrders = [];
    renderScheduledOrders();
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
    loadScheduledOrdersForStudent(student.id);
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
        const response = await fetchWithAuth(`/api/students/${studentId}`);
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
        searchResults.classList.add('visible');
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
            searchResults.classList.remove('visible');
        });
        searchResults.appendChild(div);
    });

    searchResults.classList.add('visible');
}

function searchProducts(query) {
    const trimmed = (query || '').trim().toLowerCase();
    if (!searchResults) return;
    searchResults.innerHTML = '';
    searchResults.classList.remove('visible');

    if (!trimmed) {
        return;
    }

    if (!products.length) {
        searchResults.innerHTML = '<p class="no-data">Sin productos cargados</p>';
        searchResults.classList.add('visible');
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

    const fulfilledItems = cart.map(item => ({ ...item }));
    try {
        for (const item of cart) {
            const product = products.find(p => p.id === item.productId);
            if (!product) continue;
            for (let i = 0; i < item.quantity; i++) {
                const response = await fetchWithAuth('/api/charge', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        student_id: currentStudent.id,
                        product_id: product.id,
                        point_of_sale_id: getActivePointOfSaleId(),
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
        showDispatchPanel(fulfilledItems);
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
    manualChargeBtn?.addEventListener('click', openManualChargeModal);
    manualCashInput?.addEventListener('input', handleManualCashInput);
    manualChargeCancel?.addEventListener('click', closeManualChargeModal);
    manualChargeConfirm?.addEventListener('click', confirmManualCharge);
    manualExactBtn?.addEventListener('click', () => {
        manualCashInput.value = currentCartTotal;
        handleManualCashInput();
    });
    saleClearBtn?.addEventListener('click', clearSaleState);
    dispatchClearBtn?.addEventListener('click', handleDispatchConfirm);
}

async function handleDispatchConfirm() {
    if (dispatchContext?.type === 'scheduled') {
        try {
            await dispatchScheduledOrder(dispatchContext.orderId);
            hideDispatchPanel();
        } catch (error) {
            console.error('[sales] Error despachando pedido programado', error);
        }
    } else {
        hideDispatchPanel();
        window.location.href = '/';
    }
}

async function initSalesView() {
    if (!ensureAuthenticated()) {
        return;
    }
    await ensureSalesProfile();
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
