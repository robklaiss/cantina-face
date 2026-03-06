// Backend admin panel logic
const HOST_APP_CONFIG = window.CANTINA_BACKEND_CONFIG
    || window.CANTINA_FACE_CONFIG
    || {};

const API_BASE = (HOST_APP_CONFIG.API_BASE || '').replace(/\/$/, '');

const withApiBase = (path = '') => {
    if (!path) return API_BASE;
    if (/^https?:\/\//i.test(path)) return path;
    if (!API_BASE) return path;
    return `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;
};

const API = {
    login: '/auth.php?action=login',
    users: '/backend.php?action=users',
    resetPassword: (id) => `/backend.php?action=user_reset&id=${id}`,
    stockAlerts: '/backend.php?action=stats',
    minStock: (productId) => `/backend.php?action=min_stock&product_id=${productId}`,
    linkRequests: '/backend.php?action=link_requests',
    approveLinkRequest: (id) => `/backend.php?action=link_request_decision&id=${id}`,
    rejectLinkRequest: (id) => `/backend.php?action=link_request_decision&id=${id}`,
    topups: '/backend.php?action=topups',
    approveTopup: (id) => `/backend.php?action=topup_decision&id=${id}`,
    rejectTopup: (id) => `/backend.php?action=topup_decision&id=${id}`,
    students: '/students.php?action=list',
    triggerUpdate: '/backend.php?action=trigger_update',
    updateStatus: '/backend.php?action=update_status',
};

let authToken = null;
let usersCache = [];
let alertsCache = [];
let linkRequestsCache = [];
let topupsCache = [];
let studentsCache = [];
let supportsUserDetailEndpoint = true;

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
const reloadStudentsBtn = document.getElementById('reload-students');
const studentsTableBody = document.getElementById('students-table-body');
const backendModal = document.getElementById('backend-modal');
const modalBody = document.getElementById('modal-body');
const modalTitle = document.getElementById('modal-title');
const closeModalBtn = document.getElementById('close-modal');
const notifications = document.getElementById('backend-notifications');
const checkCajaStatusBtn = document.getElementById('check-caja-status');
const triggerCajaUpdateBtn = document.getElementById('trigger-caja-update');
const cajaStatusDisplay = document.getElementById('caja-status-display');

function statusLabel(status) {
    const map = {
        pending: 'Pendiente',
        approved: 'Aprobado',
        rejected: 'Rechazado',
    };
    return map[status] || (status ? status.charAt(0).toUpperCase() + status.slice(1) : '');
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function openLinkRequestApprovalModal(requestId) {
    const request = linkRequestsCache.find((r) => String(r.id) === String(requestId));
    if (!request || !backendModal || !modalBody || !modalTitle) return;

    const suggestedStudentId = String(
        request.student_code || request.student_external_id || request.student_identifier || ''
    ).trim();
    const infoRows = [
        { label: 'Alumno solicitado', value: request.student_name || '—' },
        { label: 'Grado', value: request.student_grade || '—' },
        { label: 'Parent ID', value: request.parent_id || '—' },
        { label: 'Student identifier', value: request.student_identifier || '—' },
        { label: 'Student code', value: request.student_code || '—' },
        { label: 'Student external ID', value: request.student_external_id || '—' },
    ];

    modalTitle.textContent = 'Aprobar solicitud de vínculo';
    modalBody.innerHTML = `
        <form id="link-request-approve-form">
            <div class="modal-info-list">
                ${infoRows.map((row) => `
                    <div class="modal-info-row">
                        <span class="modal-info-label">${escapeHtml(row.label)}</span>
                        <strong class="modal-info-value">${escapeHtml(row.value)}</strong>
                    </div>
                `).join('')}
            </div>
            <div>
                <label for="link-request-student-id">ID del alumno</label>
                <input
                    id="link-request-student-id"
                    type="text"
                    name="student_id"
                    value="${escapeHtml(suggestedStudentId)}"
                    placeholder="ID interno, external_id o identifier"
                    required
                >
            </div>
            <div class="modal-actions">
                <button type="submit" class="primary-btn">Aprobar</button>
                <button type="button" class="ghost-btn" id="cancel-link-request-approve">Cancelar</button>
            </div>
        </form>
    `;

    backendModal.classList.remove('hidden');

    const form = document.getElementById('link-request-approve-form');
    const input = document.getElementById('link-request-student-id');
    const cancelBtn = document.getElementById('cancel-link-request-approve');

    cancelBtn?.addEventListener('click', () => backendModal.classList.add('hidden'));
    input?.focus();
    input?.select();

    form?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitBtn = form.querySelector('button[type="submit"]');
        const studentId = String(new FormData(form).get('student_id') || '').trim();

        if (!studentId) {
            showToast('Debes ingresar el ID del alumno para aprobar la solicitud.', 'error');
            input?.focus();
            return;
        }

        if (submitBtn) submitBtn.disabled = true;
        try {
            await apiFetch(API.approveLinkRequest(requestId), {
                method: 'POST',
                body: JSON.stringify({ decision: 'approved', student_id: studentId }),
            });
            showToast('Solicitud aprobada', 'success');
            backendModal.classList.add('hidden');
            await loadLinkRequests();
            await loadUsers();
        } catch (error) {
            showToast(error.message, 'error');
            if (submitBtn) submitBtn.disabled = false;
        }
    });
}

function openLinkRequestRejectModal(requestId) {
    const request = linkRequestsCache.find((r) => String(r.id) === String(requestId));
    if (!request || !backendModal || !modalBody || !modalTitle) return;

    modalTitle.textContent = 'Rechazar solicitud de vínculo';
    modalBody.innerHTML = `
        <form id="link-request-reject-form">
            <div class="modal-info-list compact">
                <div class="modal-info-row">
                    <span class="modal-info-label">Alumno solicitado</span>
                    <strong class="modal-info-value">${escapeHtml(request.student_name || '—')}</strong>
                </div>
                <div class="modal-info-row">
                    <span class="modal-info-label">Parent ID</span>
                    <strong class="modal-info-value">${escapeHtml(request.parent_id || '—')}</strong>
                </div>
            </div>
            <div>
                <label for="link-request-admin-notes">Observación / motivo (opcional)</label>
                <textarea id="link-request-admin-notes" name="admin_notes" rows="4" placeholder="Motivo del rechazo"></textarea>
            </div>
            <div class="modal-actions">
                <button type="submit" class="ghost-btn danger">Rechazar</button>
                <button type="button" class="secondary-btn" id="cancel-link-request-reject">Cancelar</button>
            </div>
        </form>
    `;

    backendModal.classList.remove('hidden');

    const form = document.getElementById('link-request-reject-form');
    const textarea = document.getElementById('link-request-admin-notes');
    const cancelBtn = document.getElementById('cancel-link-request-reject');

    cancelBtn?.addEventListener('click', () => backendModal.classList.add('hidden'));
    textarea?.focus();

    form?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitBtn = form.querySelector('button[type="submit"]');
        const adminNotes = String(new FormData(form).get('admin_notes') || '').trim() || null;

        if (submitBtn) submitBtn.disabled = true;
        try {
            await apiFetch(API.rejectLinkRequest(requestId), {
                method: 'POST',
                body: JSON.stringify({ decision: 'rejected', admin_notes: adminNotes }),
            });
            showToast('Solicitud rechazada', 'info');
            backendModal.classList.add('hidden');
            await loadLinkRequests();
            await loadUsers();
        } catch (error) {
            showToast(error.message, 'error');
            if (submitBtn) submitBtn.disabled = false;
        }
    });
}

async function loadStudents() {
    if (!studentsTableBody) return;
    try {
        studentsTableBody.innerHTML = '<tr><td colspan="5" class="muted">Cargando alumnos...</td></tr>';
        const students = await apiFetch(API.students);
        const studentsArray = Array.isArray(students) ? students : [];
        studentsCache = studentsArray;
        renderStudents(studentsArray);
    } catch (error) {
        showToast(`Error cargando alumnos: ${error.message}`, 'error');
        studentsTableBody.innerHTML = '<tr><td colspan="5" class="muted">No se pudieron cargar los alumnos.</td></tr>';
    }
}

function renderStudents(students = []) {
    if (!studentsTableBody) return;
    if (!students.length) {
        studentsTableBody.innerHTML = '<tr><td colspan="5" class="muted">No hay alumnos cargados</td></tr>';
        return;
    }

    studentsTableBody.innerHTML = '';
    students.forEach((student) => {
        const internalId = student.id;
        const externalId = student.external_id || null;
        const displayId = externalId || internalId;
        const copyValue = externalId || String(internalId);

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><code>${displayId}</code></td>
            <td>${student.name || '—'}</td>
            <td>${student.grade || '—'}</td>
            <td>${formatCurrencyGs(student.balance)}</td>
            <td class="actions">
                <button class="ghost-btn" data-student-copy="${copyValue}">Copiar ID</button>
            </td>
        `;
        studentsTableBody.appendChild(row);
    });

    studentsTableBody.querySelectorAll('button[data-student-copy]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const value = btn.dataset.studentCopy;
            navigator.clipboard?.writeText(value).then(() => {
                showToast(`ID copiado: ${value}`, 'success');
            }).catch(() => showToast('No se pudo copiar el ID', 'error'));
        });
    });
}

function openTopupDetails(topupId) {
    const topup = topupsCache.find((t) => String(t.id) === String(topupId));
    if (!topup || !backendModal || !modalBody || !modalTitle) return;

    const parentInfo = [
        `<p><strong>Responsable:</strong> ${topup.parent_name || 'Nombre no disponible'}</p>`,
        topup.parent_email ? `<p><strong>Email:</strong> ${topup.parent_email}</p>` : '',
        topup.parent_phone ? `<p><strong>Teléfono:</strong> ${topup.parent_phone}</p>` : '',
    ].join('');

    const allocList = (topup.allocation_details || []).map((a) => `
        <li>
            <strong>${a.student_name || a.student_id}${a.student_grade ? ` (${a.student_grade})` : ''}:</strong> ${formatCurrencyGs(a.amount)} Gs
        </li>
    `).join('');

    modalTitle.textContent = `Solicitud #${topup.id}`;
    modalBody.innerHTML = `
        <p class="muted">Estado: <span class="chip ${topup.status}">${statusLabel(topup.status)}</span></p>
        <hr class="divider" />
        <details>
            <summary>Contacto del padre/madre</summary>
            ${parentInfo || '<p class="muted">Sin datos de contacto</p>'}
        </details>
        <hr class="divider" />
        ${topup.payment_reference ? `<p><strong>Referencia pago:</strong> ${topup.payment_reference}</p>` : ''}
        <p><strong>Monto total:</strong> ${formatCurrencyGs(topup.total_amount)} Gs</p>
        <p><strong>Modo:</strong> ${topup.allocation_mode === 'custom' ? 'Personalizado' : 'Distribución equitativa'}</p>
        <hr class="divider" />
        <div class="allocations-block">
            <p class="muted">Destinatarios:</p>
            <ul>${allocList || '<li class="muted">No disponible</li>'}</ul>
        </div>
    `;
    backendModal.classList.remove('hidden');
}

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
        const requestsArray = Array.isArray(requests) ? requests : [];
        linkRequestsCache = requestsArray;
        renderLinkRequests(requestsArray);
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
                <span class="chip ${request.status}">${statusLabel(request.status)}</span>
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
    if (action === 'approve') {
        openLinkRequestApprovalModal(requestId);
        return;
    }

    openLinkRequestRejectModal(requestId);
}

async function loadTopups() {
    if (!topupsList) return;
    try {
        topupsList.innerHTML = '<p class="muted">Cargando solicitudes...</p>';
        const topups = await apiFetch(API.topups);
        const topupsArray = Array.isArray(topups) ? topups : [];
        topupsCache = topupsArray;
        renderTopups(topupArraySafe(topupsArray));
    } catch (error) {
        showToast(`Error cargando cargas de saldo: ${error.message}`, 'error');
        topupsList.innerHTML = '<p class="muted">No se pudieron cargar las solicitudes.</p>';
    }
}

function topupArraySafe(arr) {
    return Array.isArray(arr) ? arr : [];
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
        const allocationsHtml = (topup.allocation_details || []).map((a) => `
            <li><strong>${formatCurrencyGs(a.amount)} Gs</strong> • ${a.student_name || a.student_id}${a.student_grade ? ` (${a.student_grade})` : ''}</li>
        `).join('');
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
                    <small>Solicitante: ${topup.parent_name || 'Nombre no disponible'}</small>
                </div>
                <span class="chip ${topup.status}">${statusLabel(topup.status)}</span>
            </header>
            <p class="muted">Monto total: <strong>${formatCurrencyGs(topup.total_amount)} Gs</strong></p>
            <p class="muted">Modo: ${topup.allocation_mode === 'custom' ? 'Personalizado' : 'Distribución equitativa'}</p>
            ${topup.payment_reference ? `<p class="note">Referencia: ${topup.payment_reference}</p>` : ''}
            <hr class="divider" />
            <p class="muted"><strong>Solicitante:</strong> ${topup.parent_name || 'Nombre no disponible'}</p>
            ${topup.parent_email ? `<p class="muted">Email: ${topup.parent_email}</p>` : ''}
            ${topup.parent_phone ? `<p class="muted">Teléfono: ${topup.parent_phone}</p>` : ''}
            <hr class="divider" />
            <div class="allocations-block">
                <p class="muted">Asignaciones:</p>
                <ul>${allocationsHtml || '<li class="muted">No disponible</li>'}</ul>
            </div>
            <footer>
                ${footerActions}
                <button class="ghost-btn" data-topup="${topup.id}" data-action="details">Ver detalles</button>
            </footer>
        `;
        topupsList.appendChild(card);
    });

    topupsList.querySelectorAll('button[data-topup]').forEach((btn) => {
        const action = btn.dataset.action;
        if (action === 'change') {
            btn.addEventListener('click', () => openTopupStateModal(btn.dataset.topup));
        } else if (action === 'details') {
            btn.addEventListener('click', () => openTopupDetails(btn.dataset.topup));
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
                body: JSON.stringify({
                    decision: 'approved',
                    payment_reference: reference
                }),
            });
            showToast('Top-up aprobado', 'success');
        } else {
            const reference = prompt('Referencia/nota para rechazo (opcional):', topup.payment_reference || '') || topup.payment_reference || null;
            await apiFetch(API.rejectTopup(topupId), {
                method: 'POST',
                body: JSON.stringify({
                    decision: 'rejected',
                    payment_reference: reference
                }),
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

async function loadToken() {
    try {
        const response = await fetch(withApiBase('/auth.php?action=me'), {
            credentials: 'same-origin'
        });
        if (response.ok) {
            const data = await response.json();
            const user = data.user || data.data?.user;
            if (user && user.email) {
                if (currentUserChip) currentUserChip.textContent = user.email;
                localStorage.setItem('cantina_backend_user', user.email);
                loginView?.classList.add('hidden');
                appView?.classList.remove('hidden');
                refreshData();
            }
        }
    } catch (err) {
        console.log('No active session');
    }
}

async function apiFetch(url, options = {}) {
    const headers = options.headers || {};
    headers['Content-Type'] = headers['Content-Type'] || 'application/json';

    const response = await fetch(withApiBase(url), { ...options, headers, credentials: 'same-origin' });
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
            detail = data.detail || data.error || JSON.stringify(data);
        } catch (_) {
            /* ignore */
        }
        const error = new Error(detail);
        error.status = response.status;
        throw error;
    }

    const json = await response.json();
    return json.data !== undefined ? json.data : json;
}

loginForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (loginError) loginError.textContent = '';
    const formData = new FormData(loginForm);
    const payload = new URLSearchParams();
    formData.forEach((value, key) => payload.append(key, value));

    try {
        const result = await fetch(withApiBase(API.login), {
            method: 'POST',
            body: payload,
        }).then((res) => {
            if (!res.ok) throw new Error('Credenciales incorrectas');
            return res.json();
        });

        const email = formData.get('username');
        if (result.access_token) {
            saveToken(result.access_token, email);
        }
        loginView?.classList.add('hidden');
        appView?.classList.remove('hidden');
        if (currentUserChip) currentUserChip.textContent = email;
        showToast('Sesión iniciada', 'success');
        refreshData();
    } catch (error) {
        if (loginError) loginError.textContent = error.message || 'Error al iniciar sesión';
    }
});

function logout() {
    localStorage.removeItem('cantina_backend_token');
    localStorage.removeItem('cantina_backend_user');
    authToken = null;
    appView?.classList.add('hidden');
    loginView?.classList.remove('hidden');
    loginForm?.reset();
    showToast('Sesión cerrada', 'info');
}

logoutBtn?.addEventListener('click', logout);
refreshDataBtn?.addEventListener('click', refreshData);
reloadAlertsBtn?.addEventListener('click', loadAlerts);
reloadLinkRequestsBtn?.addEventListener('click', loadLinkRequests);
reloadTopupsBtn?.addEventListener('click', loadTopups);
reloadStudentsBtn?.addEventListener('click', loadStudents);
createUserBtn?.addEventListener('click', () => openUserModal());
closeModalBtn?.addEventListener('click', () => backendModal?.classList.add('hidden'));
backendModal?.addEventListener('click', (e) => {
    if (e.target === backendModal) backendModal.classList.add('hidden');
});
checkCajaStatusBtn?.addEventListener('click', checkCajaStatus);
triggerCajaUpdateBtn?.addEventListener('click', triggerCajaUpdate);

async function refreshData() {
    await Promise.all([loadUsers(), loadAlerts(), loadLinkRequests(), loadTopups(), loadStudents()]).catch((err) => {
        console.error(err);
    });
}

async function loadUsers() {
    try {
        const users = await apiFetch(API.users);
        const usersArray = Array.isArray(users) ? users : [];
        usersCache = usersArray;
        renderUsers(usersArray);
        updateUserStats(usersArray);
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
        const alertsArray = Array.isArray(alerts) ? alerts : [];
        alertsCache = alertsArray;
        renderAlerts(alertsArray);
        if (statStockAlerts) {
            const affected = alertsArray.filter((a) => a.status !== 'ok').length;
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

async function openUserModal(userId = null) {
    backendModal?.classList.remove('hidden');
    let user = usersCache.find((u) => u.id === userId);

    if (userId && (!user || !user.hashed_password) && supportsUserDetailEndpoint) {
        try {
            user = await apiFetch(`${API.users}/${userId}`);
            const exists = usersCache.some((u) => u.id === user.id);
            usersCache = exists
                ? usersCache.map((u) => (u.id === user.id ? user : u))
                : [...usersCache, user];
        } catch (error) {
            if (error?.status === 404) {
                supportsUserDetailEndpoint = false;
                showToast('Detalle de usuario no disponible en el servidor actual. Se mostrarán los datos en caché.', 'info');
            } else {
                showToast(`No se pudo cargar el usuario: ${error.message}`, 'error');
                backendModal?.classList.add('hidden');
                return;
            }
        }
    }

    if (!modalTitle || !modalBody) return;

    modalTitle.textContent = user ? 'Editar usuario' : 'Nuevo usuario';
    const currentPasswordValue = user?.hashed_password ?? 'No disponible';

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
            ${user ? `
            <div>
                <label>Contraseña actual</label>
                <input type="text" value="${currentPasswordValue}" disabled>
            </div>` : ''}
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
            backendModal?.classList.add('hidden');
            refreshData();
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
}

function promptResetPassword(userId) {
    const user = usersCache.find((u) => u.id === userId);
    if (!user || !modalTitle || !modalBody) return;

    backendModal?.classList.remove('hidden');
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
            backendModal?.classList.add('hidden');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
}

function openMinStockModal(productId, minStock) {
    if (!modalTitle || !modalBody) return;
    backendModal?.classList.remove('hidden');
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
            backendModal?.classList.add('hidden');
            loadAlerts();
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
}

async function checkCajaStatus() {
    if (!cajaStatusDisplay) return;
    try {
        cajaStatusDisplay.innerHTML = '<p class="muted">Consultando estado de la máquina caja...</p>';
        const status = await apiFetch(API.updateStatus);
        renderCajaStatus(status);
    } catch (error) {
        cajaStatusDisplay.innerHTML = `<p class="error-text">Error: ${error.message}</p>`;
        showToast(`Error al consultar estado: ${error.message}`, 'error');
    }
}

function renderCajaStatus(status) {
    if (!cajaStatusDisplay) return;

    const version = status.current_version || 'Desconocida';
    const lastCheck = status.last_check || 'Nunca';
    const updateAvailable = status.update_available || false;
    const remoteVersion = status.remote_version || 'N/A';

    let html = `
        <div class="caja-status-info">
            <p><strong>Versión actual:</strong> ${version}</p>
            <p><strong>Última verificación:</strong> ${lastCheck}</p>
    `;

    if (updateAvailable) {
        html += `<p class="warning"><strong>⚠️ Actualización disponible:</strong> ${remoteVersion}</p>`;
    } else {
        html += `<p class="success"><strong>✓ Sistema actualizado</strong></p>`;
    }

    html += '</div>';
    cajaStatusDisplay.innerHTML = html;
}

async function triggerCajaUpdate() {
    if (!confirm('¿Estás seguro de que deseas actualizar la máquina caja? Esto puede tomar varios minutos y la máquina no estará disponible durante el proceso.')) {
        return;
    }

    try {
        if (triggerCajaUpdateBtn) {
            triggerCajaUpdateBtn.disabled = true;
            triggerCajaUpdateBtn.textContent = '⏳ Actualizando...';
        }

        const result = await apiFetch(API.triggerUpdate, {
            method: 'POST',
            body: JSON.stringify({}),
        });

        showToast(result.message || 'Actualización iniciada correctamente', 'success');

        if (cajaStatusDisplay) {
            cajaStatusDisplay.innerHTML = `
                <div class="caja-status-info">
                    <p class="success"><strong>✓ Actualización iniciada</strong></p>
                    <p class="muted">La máquina caja se está actualizando. Este proceso puede tomar varios minutos.</p>
                    <p class="muted">Verifica el estado en unos minutos para confirmar que la actualización se completó.</p>
                </div>
            `;
        }

        setTimeout(() => {
            if (triggerCajaUpdateBtn) {
                triggerCajaUpdateBtn.disabled = false;
                triggerCajaUpdateBtn.textContent = '🔄 Actualizar Máquina Caja';
            }
        }, 5000);

    } catch (error) {
        showToast(`Error al actualizar: ${error.message}`, 'error');
        if (triggerCajaUpdateBtn) {
            triggerCajaUpdateBtn.disabled = false;
            triggerCajaUpdateBtn.textContent = '🔄 Actualizar Máquina Caja';
        }
    }
}

window.addEventListener('DOMContentLoaded', loadToken);