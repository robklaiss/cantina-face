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
    linkRequests: '/backend.php?action=link_requests',
};

const currentUserChip = document.getElementById('current-user-chip');
const notifications = document.getElementById('backend-notifications');
const reloadLinksBtn = document.getElementById('reload-links');
const linksSearchInput = document.getElementById('links-search-input');
const linksStatusFilter = document.getElementById('links-status-filter');
const linksTableBody = document.getElementById('links-table-body');

const state = {
    links: [],
    filters: {
        query: '',
        status: '',
    },
};

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

function statusLabel(status) {
    const map = {
        pending: 'Pendiente',
        approved: 'Aprobado',
        rejected: 'Rechazado',
    };
    return map[status] || (status ? status.charAt(0).toUpperCase() + status.slice(1) : '—');
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
            
        }
        throw new Error(detail);
    }

    const json = await response.json();
    return json.data !== undefined ? json.data : json;
}

function getFilteredLinks() {
    const query = String(state.filters.query || '').trim().toLowerCase();
    const selectedStatus = String(state.filters.status || '').trim();

    return state.links.filter((link) => {
        const haystack = [
            link.status,
            link.parent_id,
            link.parent_name,
            link.student_name,
            link.student_grade,
            link.student_identifier,
            link.student_internal_id,
            link.student_external_id,
            link.student_code,
        ]
            .map((value) => String(value || '').toLowerCase())
            .join(' ');

        const matchesQuery = !query || haystack.includes(query);
        const matchesStatus = !selectedStatus || String(link.status || '') === selectedStatus;
        return matchesQuery && matchesStatus;
    });
}

function renderLinks(rows = []) {
    if (!linksTableBody) return;
    if (!rows.length) {
        linksTableBody.innerHTML = '<tr><td colspan="10" class="muted">No se encontraron vínculos con los filtros actuales.</td></tr>';
        return;
    }

    linksTableBody.innerHTML = '';
    rows.forEach((link) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><span class="chip ${escapeHtml(link.status || '')}">${escapeHtml(statusLabel(link.status))}</span></td>
            <td><span class="mono-text">${escapeHtml(link.parent_id ?? '—')}</span></td>
            <td>${escapeHtml(link.parent_name || '—')}</td>
            <td>${escapeHtml(link.student_name || '—')}</td>
            <td>${escapeHtml(link.student_grade || '—')}</td>
            <td>${escapeHtml(link.student_identifier || '—')}</td>
            <td><span class="mono-text">${escapeHtml(link.student_internal_id || '—')}</span></td>
            <td><span class="mono-text">${escapeHtml(link.student_external_id || link.student_code || '—')}</span></td>
            <td>${escapeHtml(formatDateTime(link.created_at))}</td>
            <td>${escapeHtml(formatDateTime(link.processed_at))}</td>
        `;
        linksTableBody.appendChild(row);
    });
}

function applyFilters() {
    renderLinks(getFilteredLinks());
}

async function loadLinks() {
    if (!linksTableBody) return;
    try {
        linksTableBody.innerHTML = '<tr><td colspan="10" class="muted">Cargando vínculos...</td></tr>';
        const links = await apiFetch(API.linkRequests);
        state.links = Array.isArray(links) ? links : [];
        applyFilters();
    } catch (error) {
        showToast(error.message || 'No se pudieron cargar los vínculos', 'error');
        linksTableBody.innerHTML = '<tr><td colspan="10" class="muted">No se pudieron cargar los vínculos.</td></tr>';
    }
}

async function bootstrapLinksPage() {
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
        await loadLinks();
    } catch (error) {
        showToast(error.message || 'No se pudo cargar la página de vínculos', 'error');
    }
}

linksSearchInput?.addEventListener('input', () => {
    state.filters.query = linksSearchInput.value || '';
    applyFilters();
});

linksStatusFilter?.addEventListener('change', () => {
    state.filters.status = linksStatusFilter.value || '';
    applyFilters();
});

reloadLinksBtn?.addEventListener('click', loadLinks);
window.addEventListener('DOMContentLoaded', bootstrapLinksPage);
