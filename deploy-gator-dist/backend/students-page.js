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
    me: '/auth.php?action=me',
    students: '/students.php?action=list',
    studentLinkDetails: (id) => `/students.php?action=link_details&id=${encodeURIComponent(id)}`,
};

const currentUserChip = document.getElementById('current-user-chip');
const notifications = document.getElementById('backend-notifications');
const reloadStudentsBtn = document.getElementById('reload-students');
const studentsTableBody = document.getElementById('students-table-body');
const studentsSearchInput = document.getElementById('students-search-input');
const studentsGradeFilter = document.getElementById('students-grade-filter');
const backendModal = document.getElementById('backend-modal');
const modalBody = document.getElementById('modal-body');
const modalTitle = document.getElementById('modal-title');
const closeModalBtn = document.getElementById('close-modal');

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function showToast(message, type = 'info') {
    if (!notifications) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    notifications.appendChild(toast);
    setTimeout(() => toast.remove(), 4500);
}

function formatCurrencyGs(value) {
    const amount = Number(value) || 0;
    try {
        return amount.toLocaleString('es-PY', { maximumFractionDigits: 0 });
    } catch (_) {
        return amount.toString();
    }
}

function formatDateTime(value) {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return String(value);
    }
    try {
        return date.toLocaleString('es-PY');
    } catch (_) {
        return date.toISOString();
    }
}

async function apiFetch(url, options = {}) {
    const headers = options.headers || {};
    headers['Content-Type'] = headers['Content-Type'] || 'application/json';

    const response = await fetch(withApiBase(url), { ...options, headers, credentials: 'same-origin' });
    if (response.status === 401) {
        window.location.href = 'index.html';
        throw new Error('Sesión expirada');
    }

    if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
            const data = await response.json();
            detail = data.detail || data.error || JSON.stringify(data);
        } catch (_) {
            /* ignore */
        }
        throw new Error(detail);
    }

    const json = await response.json();
    return json.data !== undefined ? json.data : json;
}

async function openStudentLinksModal(studentId) {
    if (!studentId || !backendModal || !modalBody || !modalTitle) return;

    modalTitle.textContent = 'Vínculos del alumno';
    modalBody.innerHTML = '<p class="muted">Cargando vínculos...</p>';
    backendModal.classList.remove('hidden');

    try {
        const payload = await apiFetch(API.studentLinkDetails(studentId));
        const student = payload?.student || {};
        const links = Array.isArray(payload?.links) ? payload.links : [];
        const infoRows = [
            { label: 'Alumno', value: student.name || '—' },
            { label: 'Grado', value: student.grade || '—' },
            { label: 'ID interno', value: student.id || '—' },
            { label: 'ID externo', value: student.external_id || '—' },
        ];

        const linksHtml = links.length
            ? `
                <ul class="student-links-list">
                    ${links.map((link) => `
                        <li class="student-link-card">
                            <h4>${escapeHtml(link.parent_name || `Tutor #${link.parent_id || ''}`)}</h4>
                            <p><strong>Parent ID:</strong> <span class="mono-text">${escapeHtml(link.parent_id ?? '—')}</span></p>
                            ${link.parent_email ? `<p><strong>Email:</strong> ${escapeHtml(link.parent_email)}</p>` : ''}
                            ${link.parent_phone ? `<p><strong>Teléfono:</strong> ${escapeHtml(link.parent_phone)}</p>` : ''}
                            ${link.parent_dni ? `<p><strong>DNI:</strong> ${escapeHtml(link.parent_dni)}</p>` : ''}
                            <p><strong>Vinculado desde:</strong> ${escapeHtml(formatDateTime(link.linked_at))}</p>
                        </li>
                    `).join('')}
                </ul>
            `
            : '<p class="muted">Este alumno todavía no tiene vínculos activos.</p>';

        modalTitle.textContent = `Vínculos de ${student.name || 'alumno'}`;
        modalBody.innerHTML = `
            <div class="modal-info-list compact">
                ${infoRows.map((row) => `
                    <div class="modal-info-row">
                        <span class="modal-info-label">${escapeHtml(row.label)}</span>
                        <strong class="modal-info-value">${escapeHtml(row.value)}</strong>
                    </div>
                `).join('')}
            </div>
            <div style="margin-top: 16px;">
                ${linksHtml}
            </div>
        `;
    } catch (error) {
        modalBody.innerHTML = `<p class="error-text">${escapeHtml(error.message || 'No se pudieron cargar los vínculos del alumno')}</p>`;
    }
}

const studentsDirectory = window.CantinaStudentsDirectory?.createStudentsDirectory({
    apiFetch,
    apiPath: API.students,
    formatCurrencyGs,
    notify: showToast,
    notifications,
    onViewLinks: openStudentLinksModal,
    reloadButton: reloadStudentsBtn,
    searchInput: studentsSearchInput,
    gradeFilter: studentsGradeFilter,
    tableBody: studentsTableBody,
    tableColspan: 6,
});

closeModalBtn?.addEventListener('click', () => backendModal?.classList.add('hidden'));
backendModal?.addEventListener('click', (event) => {
    if (event.target === backendModal) {
        backendModal.classList.add('hidden');
    }
});

async function bootstrapStudentsPage() {
    try {
        const response = await fetch(withApiBase(API.me), { credentials: 'same-origin' });
        if (!response.ok) {
            window.location.href = 'index.html';
            return;
        }
        const data = await response.json();
        const user = data.user || data.data?.user;
        if (!user?.email) {
            window.location.href = 'index.html';
            return;
        }
        if (currentUserChip) currentUserChip.textContent = user.email;
        await studentsDirectory?.load();
    } catch (error) {
        showToast(error.message || 'No se pudo cargar la página de alumnos', 'error');
    }
}

window.addEventListener('DOMContentLoaded', bootstrapStudentsPage);
