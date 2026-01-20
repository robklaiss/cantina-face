// Backend admin panel logic
const API = {
    login: '/auth/token',
    users: '/auth/users',
    resetPassword: (id) => `/auth/users/${id}/reset-password`,
    stockAlerts: '/api/stock/alerts',
    minStock: (productId) => `/api/products/${productId}/min-stock`,
    linkRequests: '/api/link-requests',
    approveLinkRequest: (id) => `/api/link-requests/${id}/approve`,
    rejectLinkRequest: (id) => `/api/link-requests/${id}/reject`,
    topups: '/api/topups',
    approveTopup: (id) => `/api/topups/${id}/approve`,
    rejectTopup: (id) => `/api/topups/${id}/reject`,
};

let authToken = null;
let usersCache = [];
let alertsCache = [];
let linkRequestsCache = [];
let topupsCache = [];

const loginView = document.getElementById('backend-login');
const appView = document.getElementById('backend-app');
const loginForm = document.getElementById('login-form');
const loginError = document.getElementById('login-error');
const statTotalUsers = document.getElementById('stat-total-users');
const statCashiers = document.getElementById('stat-cashiers');
const statStockAlerts = document.getElementById('stat-stock-alerts');
const currentUserChip = document.getElementById('current-user-chip');
const usersTableBody = document.getElementById('users-table-body');
const alertsGrid = document.getElementById('alerts-grid');
const refreshDataBtn = document.getElementById('refresh-data-btn');
const reloadAlertsBtn = document.getElementById('reload-alerts-btn');
const logoutBtn = document.getElementById('logout-btn');
const createUserBtn = document.getElementById('open-create-user');
const linkRequestsList = document.getElementById('link-requests-list');
const reloadLinkRequestsBtn = document.getElementById('reload-link-requests');
const topupsList = document.getElementById('topups-list');
const reloadTopupsBtn = document.getElementById('reload-topups');
const backendModal = document.getElementById('backend-modal');
const modalBody = document.getElementById('modal-body');
const modalTitle = document.getElementById('modal-title');
const closeModalBtn = document.getElementById('close-modal');
const notifications = document.getElementById('backend-notifications');

function showToast(message, type = 'info') {
    if (!notifications) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    notifications.appendChild(toast);
    setTimeout(() => toast.remove(), 4500);
}

function openTopupStateModal(topupId) {
    const topup = topupsCache.find((t) => String(t.id) === String(topupId));
    if (!topup || !backendModal || !modalBody || !modalTitle) return;

    modalTitle.textContent = `Solicitud #${topup.id}`;
    modalBody.innerHTML = `
        <p class="muted">Estado actual: <strong>${topup.status.toUpperCase()}</strong></p>
        <div class="modal-actions">
            <button class="primary-btn" id="edit-topup-btn">Editar</button>
            <button class="ghost-btn danger" id="suspend-topup-btn">Suspender</button>
        </div>
        <p class="muted" style="margin-top:12px; font-size:0.9rem;">
            Estas acciones requieren implementación adicional en el backend.
        </p>
    `;

    backendModal.classList.remove('hidden');

    const editBtn = document.getElementById('edit-topup-btn');
    const suspendBtn = document.getElementById('suspend-topup-btn');
    editBtn?.addEventListener('click', () => {
        showToast('La edición de top-ups aún no está disponible.', 'info');
    });
    suspendBtn?.addEventListener('click', () => {
        showToast('La suspensión de top-ups aún no está disponible.', 'info');
    });
}

function formatCurrencyGs(value) {
    const amount = Number(value) || 0;
    try {
        return amount.toLocaleString('es-PY', { maximumFractionDigits: 0 });
    } catch (_) {
        return amount.toString();
    }
}

async function loadLinkRequests() {
    if (!linkRequestsList) return;
    try {
        linkRequestsList.innerHTML = '<p class="muted">Cargando solicitudes...</p>';
        const requests = await apiFetch(API.linkRequests);
        linkRequestsCache = requests;
        renderLinkRequests(requests);
    } catch (error) {
        showToast(`Error cargando solicitudes: ${error.message}`, 'error');
        linkRequestsList.innerHTML = '<p class="muted">No se pudieron cargar las solicitudes.</p>';
    }
}

function renderLinkRequests(requests = []) {
    if (!linkRequestsList) return;
    if (!requests.length) {
        linkRequestsList.innerHTML = '<p class="muted">Sin solicitudes pendientes.</p>';
        return;
    }

    linkRequestsList.innerHTML = '';
    requests.forEach((request) => {
        const card = document.createElement('article');
        card.className = `request-card status-${request.status}`;
        card.innerHTML = `
            <header>
                <div>
                    <h4>${request.student_name}</h4>
                    <small>${request.student_grade || ''}</small>
                </div>
                <span class="chip ${request.status}">${request.status.toUpperCase()}</span>
            </header>
            <p class="muted">ID Tutor: ${request.parent_id}</p>
            ${request.student_identifier ? `<p class="muted">Identificador: ${request.student_identifier}</p>` : ''}
            ${request.notes ? `<p class="note">${request.notes}</p>` : ''}
            ${request.admin_notes ? `<p class="note admin">Nota admin: ${request.admin_notes}</p>` : ''}
            <footer>
                <button class="secondary-btn" data-action="approve" data-id="${request.id}">Aprobar</button>
                <button class="ghost-btn" data-action="reject" data-id="${request.id}">Rechazar</button>
            </footer>
        `;
        if (request.status !== 'pending') {
            card.querySelectorAll('button').forEach((btn) => (btn.disabled = true));
        }
        linkRequestsList.appendChild(card);
    });

    linkRequestsList.querySelectorAll('button[data-action]').forEach((btn) => {
        btn.addEventListener('click', () => handleLinkRequestAction(btn.dataset.id, btn.dataset.action));
    });
}

async function handleLinkRequestAction(requestId, action) {
    const request = linkRequestsCache.find((r) => String(r.id) === String(requestId));
    if (!request) return;
    try {
        if (action === 'approve') {
            const studentId = prompt('Ingrese el ID del alumno que corresponde a esta solicitud');
            if (!studentId) return;
            await apiFetch(API.approveLinkRequest(requestId), {
                method: 'POST',
                body: JSON.stringify({ student_id: studentId }),
            });
            showToast('Solicitud aprobada', 'success');
        } else {
            const adminNotes = prompt('Ingrese un motivo de rechazo (opcional)') || null;
            await apiFetch(API.rejectLinkRequest(requestId), {
                method: 'POST',
                body: JSON.stringify({ admin_notes: adminNotes }),
            });
            showToast('Solicitud rechazada', 'info');
        }
        await loadLinkRequests();
        await loadUsers();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

async function loadTopups() {
    if (!topupsList) return;
    try {
        topupsList.innerHTML = '<p class="muted">Cargando solicitudes...</p>';
        const topups = await apiFetch(API.topups);
        topupsCache = topups;
        renderTopups(topups);
    } catch (error) {
        showToast(`Error cargando cargas de saldo: ${error.message}`, 'error');
        topupsList.innerHTML = '<p class="muted">No se pudieron cargar las solicitudes.</p>';
    }
}

function renderTopups(topupRequests = []) {
    if (!topupsList) return;
    if (!topupRequests.length) {
        topupsList.innerHTML = '<p class="muted">Sin solicitudes pendientes.</p>';
        return;
    }

    topupsList.innerHTML = '';
    topupRequests.forEach((topup) => {
        const card = document.createElement('article');
        card.className = `request-card status-${topup.status}`;
        const allocationsHtml = Object.entries(topup.allocations || {})
            .map(([studentId, amount]) => `<li><strong>${formatCurrencyGs(amount)} Gs</strong> • Alumno ${studentId}</li>`)
            .join('');
        const isPending = topup.status === 'pending';
        const footerActions = isPending
            ? `
                <button class="secondary-btn" data-topup="${topup.id}" data-action="approve">Aprobar</button>
                <button class="ghost-btn" data-topup="${topup.id}" data-action="reject">Rechazar</button>
            `
            : `
                <button class="secondary-btn" data-topup="${topup.id}" data-action="change">Cambiar estado</button>
            `;

        card.innerHTML = `
            <header>
                <div>
                    <h4>Solicitud #${topup.id}</h4>
                    <small>Padre ID: ${topup.parent_id}</small>
                </div>
                <span class="chip ${topup.status}">${topup.status.toUpperCase()}</span>
            </header>
            <p class="muted">Monto total: <strong>${formatCurrencyGs(topup.total_amount)} Gs</strong></p>
            <p class="muted">Modo: ${topup.allocation_mode === 'custom' ? 'Personalizado' : 'Distribución equitativa'}</p>
            ${topup.payment_reference ? `<p class="note">Referencia: ${topup.payment_reference}</p>` : ''}
            <div class="allocations-block">
                <p class="muted">Asignaciones:</p>
                <ul>${allocationsHtml || '<li class="muted">No disponible</li>'}</ul>
            </div>
            <footer>
                ${footerActions}
            </footer>
        `;
        topupsList.appendChild(card);
    });

    topupsList.querySelectorAll('button[data-topup]').forEach((btn) => {
        const action = btn.dataset.action;
        if (action === 'change') {
            btn.addEventListener('click', () => openTopupStateModal(btn.dataset.topup));
        } else {
            btn.addEventListener('click', () => handleTopupAction(btn.dataset.topup, action));
        }
    });
}

async function handleTopupAction(topupId, action) {
    const topup = topupsCache.find((t) => String(t.id) === String(topupId));
    if (!topup) return;

    try {
        if (action === 'approve') {
            const reference = prompt('Referencia de pago (opcional):', topup.payment_reference || '') || topup.payment_reference || null;
            await apiFetch(API.approveTopup(topupId), {
                method: 'POST',
                body: JSON.stringify({ payment_reference: reference }),
            });
            showToast('Top-up aprobado', 'success');
        } else {
            const reference = prompt('Referencia/nota para rechazo (opcional):', topup.payment_reference || '') || topup.payment_reference || null;
            await apiFetch(API.rejectTopup(topupId), {
                method: 'POST',
                body: JSON.stringify({ payment_reference: reference }),
            });
            showToast('Top-up rechazado', 'info');
        }
        await loadTopups();
        await loadUsers();
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function saveToken(token, userEmail) {
    authToken = token;
    localStorage.setItem('cantina_backend_token', token);
    localStorage.setItem('cantina_backend_user', userEmail || '');
}

function loadToken() {
    const storedToken = localStorage.getItem('cantina_backend_token');
    const storedUser = localStorage.getItem('cantina_backend_user');
    if (storedToken) {
        authToken = storedToken;
        if (currentUserChip && storedUser) currentUserChip.textContent = storedUser;
        loginView.classList.add('hidden');
        appView.classList.remove('hidden');
        refreshData();
    }
}

async function apiFetch(url, options = {}) {
    const headers = options.headers || {};
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }
    headers['Content-Type'] = headers['Content-Type'] || 'application/json';

    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
        showToast('Sesión expirada, vuelve a ingresar.', 'error');
        localStorage.removeItem('cantina_backend_token');
        localStorage.removeItem('cantina_backend_user');
        window.location.reload();
        return Promise.reject('Unauthorized');
    }

    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
            const data = await response.json();
            detail = data.detail || JSON.stringify(data);
        } catch (_) {
            /* ignore */
        }
        throw new Error(detail);
    }

    return response.json();
}

loginForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    loginError.textContent = '';
    const formData = new FormData(loginForm);
    const payload = new URLSearchParams();
    formData.forEach((value, key) => payload.append(key, value));

    try {
        const result = await fetch(API.login, {
            method: 'POST',
            body: payload,
        }).then((res) => {
            if (!res.ok) throw new Error('Credenciales incorrectas');
            return res.json();
        });

        const email = formData.get('username');
        saveToken(result.access_token, email);
        loginView.classList.add('hidden');
        appView.classList.remove('hidden');
        if (currentUserChip) currentUserChip.textContent = email;
        showToast('Sesión iniciada', 'success');
        refreshData();
    } catch (error) {
        loginError.textContent = error.message || 'Error al iniciar sesión';
    }
});

function logout() {
    localStorage.removeItem('cantina_backend_token');
    localStorage.removeItem('cantina_backend_user');
    authToken = null;
    appView.classList.add('hidden');
    loginView.classList.remove('hidden');
    loginForm.reset();
    showToast('Sesión cerrada', 'info');
}

logoutBtn?.addEventListener('click', logout);
refreshDataBtn?.addEventListener('click', refreshData);
reloadAlertsBtn?.addEventListener('click', loadAlerts);
reloadLinkRequestsBtn?.addEventListener('click', loadLinkRequests);
reloadTopupsBtn?.addEventListener('click', loadTopups);
createUserBtn?.addEventListener('click', () => openUserModal());
closeModalBtn?.addEventListener('click', () => backendModal.classList.add('hidden'));
backendModal?.addEventListener('click', (e) => {
    if (e.target === backendModal) backendModal.classList.add('hidden');
});

async function refreshData() {
    await Promise.all([loadUsers(), loadAlerts(), loadLinkRequests(), loadTopups()]).catch((err) => {
        console.error(err);
    });
}

async function loadUsers() {
    try {
        const users = await apiFetch(API.users);
        usersCache = users;
        renderUsers(users);
        updateUserStats(users);
    } catch (error) {
        showToast(`Error cargando usuarios: ${error.message}`, 'error');
    }
}

function updateUserStats(users = []) {
    if (statTotalUsers) statTotalUsers.textContent = users.length;
    if (statCashiers) {
        const activeCashiers = users.filter((u) => u.role === 'cajera' && u.is_active).length;
        statCashiers.textContent = activeCashiers;
    }
}

function renderUsers(users = []) {
    if (!usersTableBody) return;
    if (!users.length) {
        usersTableBody.innerHTML = '<tr><td colspan="6" class="muted">No hay usuarios cargados</td></tr>';
        return;
    }

    usersTableBody.innerHTML = '';
    users.forEach((user) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${user.full_name || '—'}</td>
            <td>${user.email}</td>
            <td><span class="badge role">${user.role}</span></td>
            <td>${user.point_of_sale_id ?? '—'}</td>
            <td>
                <span class="status-pill ${user.is_active ? 'active' : 'inactive'}">
                    ${user.is_active ? 'Activo' : 'Inactivo'}
                </span>
            </td>
            <td class="actions">
                <button class="secondary-btn" data-user="${user.id}" data-action="edit">Editar</button>
                <button class="ghost-btn" data-user="${user.id}" data-action="reset">Reset pass</button>
            </td>
        `;
        usersTableBody.appendChild(row);
    });

    usersTableBody.querySelectorAll('button[data-user]').forEach((btn) => {
        btn.addEventListener('click', (e) => {
            const id = parseInt(btn.dataset.user, 10);
            const action = btn.dataset.action;
            if (action === 'edit') openUserModal(id);
            if (action === 'reset') promptResetPassword(id);
            e.stopPropagation();
        });
    });
}

async function loadAlerts() {
    try {
        const alerts = await apiFetch(API.stockAlerts);
        alertsCache = alerts;
        renderAlerts(alerts);
        if (statStockAlerts) {
            const affected = alerts.filter((a) => a.status !== 'ok').length;
            statStockAlerts.textContent = affected;
        }
    } catch (error) {
        showToast(`Error cargando alertas: ${error.message}`, 'error');
    }
}

function renderAlerts(alerts = []) {
    if (!alertsGrid) return;
    if (!alerts.length) {
        alertsGrid.innerHTML = '<p class="muted">Sin alertas de stock registradas</p>';
        return;
    }

    alertsGrid.innerHTML = '';
    alerts.forEach((alert) => {
        const statusClass = alert.status || 'ok';
        const card = document.createElement('div');
        card.className = `alert-card ${statusClass}`;
        card.innerHTML = `
            <header>
                <div>
                    <h4>${alert.product_name}</h4>
                    <small>POS: ${alert.point_of_sale_id ?? 'General'}</small>
                </div>
                <span class="badge ${statusClass}">${statusClass.toUpperCase()}</span>
            </header>
            <p class="stock-line">Stock actual: <strong>${alert.current_stock ?? 0}</strong></p>
            <p>Min. configurado: <strong>${alert.min_stock ?? '—'}</strong></p>
            <button class="secondary-btn" data-product="${alert.product_id}" data-min="${alert.min_stock ?? 0}">Editar umbral</button>
        `;
        alertsGrid.appendChild(card);
    });

    alertsGrid.querySelectorAll('button[data-product]').forEach((btn) => {
        btn.addEventListener('click', () => openMinStockModal(parseInt(btn.dataset.product, 10), parseInt(btn.dataset.min, 10)));
    });
}

function openUserModal(userId = null) {
    backendModal.classList.remove('hidden');
    const user = usersCache.find((u) => u.id === userId);
    modalTitle.textContent = user ? 'Editar usuario' : 'Nuevo usuario';

    modalBody.innerHTML = `
        <form id="user-form">
            <div>
                <label>Nombre completo</label>
                <input type="text" name="full_name" value="${user?.full_name ?? ''}" placeholder="Ej: Laura Fernández">
            </div>
            <div>
                <label>Correo electrónico</label>
                <input type="email" name="email" value="${user?.email ?? ''}" ${user ? 'disabled' : ''} required>
            </div>
            <div class="two-col">
                <div>
                    <label>Rol</label>
                    <select name="role" required>
                        ${['admin','cajera','stock','parent'].map((role) => `<option value="${role}" ${user?.role===role?'selected':''}>${role}</option>`).join('')}
                    </select>
                </div>
                <div>
                    <label>Point of Sale (opcional)</label>
                    <input type="number" name="point_of_sale_id" value="${user?.point_of_sale_id ?? ''}" min="1">
                </div>
            </div>
            <div>
                <label>Estado</label>
                <select name="is_active">
                    <option value="true" ${user?.is_active !== false ? 'selected' : ''}>Activo</option>
                    <option value="false" ${user?.is_active === false ? 'selected' : ''}>Inactivo</option>
                </select>
            </div>
            ${user ? '' : `
            <div>
                <label>Contraseña inicial</label>
                <input type="password" name="password" required placeholder="********">
            </div>`}
            <button type="submit" class="primary-btn">${user ? 'Guardar cambios' : 'Crear usuario'}</button>
        </form>
    `;

    const form = document.getElementById('user-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());
        payload.is_active = payload.is_active === 'true';
        if (payload.point_of_sale_id === '') payload.point_of_sale_id = null;
        else payload.point_of_sale_id = Number(payload.point_of_sale_id);

        try {
            if (user) {
                delete payload.email;
                delete payload.password;
                await apiFetch(`${API.users}/${user.id}`, {
                    method: 'PUT',
                    body: JSON.stringify(payload),
                });
                showToast('Usuario actualizado', 'success');
            } else {
                await apiFetch(API.users, {
                    method: 'POST',
                    body: JSON.stringify(payload),
                });
                showToast('Usuario creado', 'success');
            }
            backendModal.classList.add('hidden');
            refreshData();
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
}

function promptResetPassword(userId) {
    const user = usersCache.find((u) => u.id === userId);
    if (!user) return;

    backendModal.classList.remove('hidden');
    modalTitle.textContent = `Resetear contraseña para ${user.email}`;
    modalBody.innerHTML = `
        <form id="reset-form">
            <label>Nueva contraseña</label>
            <input type="password" name="new_password" required placeholder="********">
            <button type="submit" class="primary-btn">Resetear</button>
        </form>
    `;

    const form = document.getElementById('reset-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const payload = Object.fromEntries(new FormData(form).entries());
        try {
            await apiFetch(API.resetPassword(userId), {
                method: 'POST',
                body: JSON.stringify(payload),
            });
            showToast('Contraseña actualizada', 'success');
            backendModal.classList.add('hidden');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
}

function openMinStockModal(productId, minStock) {
    backendModal.classList.remove('hidden');
    modalTitle.textContent = 'Editar umbral de stock';
    modalBody.innerHTML = `
        <form id="min-stock-form">
            <label>Stock mínimo para alertas</label>
            <input type="number" name="min_stock" required min="0" value="${minStock ?? 0}">
            <button type="submit" class="primary-btn">Guardar umbral</button>
        </form>
    `;

    const form = document.getElementById('min-stock-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const payload = Object.fromEntries(new FormData(form).entries());
        payload.min_stock = Number(payload.min_stock);

        try {
            await apiFetch(API.minStock(productId), {
                method: 'PUT',
                body: JSON.stringify(payload),
            });
            showToast('Umbral actualizado', 'success');
            backendModal.classList.add('hidden');
            loadAlerts();
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
}

window.addEventListener('DOMContentLoaded', loadToken);
