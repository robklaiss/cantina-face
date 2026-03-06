(function attachStudentsDirectory(global) {
    function resolveElement(value) {
        if (!value) return null;
        if (typeof value === 'string') return document.getElementById(value);
        return value;
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function normalizeFilterValue(value) {
        return String(value ?? '').trim().toLowerCase();
    }

    function defaultFormatCurrencyGs(value) {
        const amount = Number(value) || 0;
        try {
            return amount.toLocaleString('es-PY', { maximumFractionDigits: 0 });
        } catch (_) {
            return amount.toString();
        }
    }

    function createToast(notifications, message, type) {
        if (!notifications) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        notifications.appendChild(toast);
        setTimeout(() => toast.remove(), 4500);
    }

    function createStudentsDirectory(options = {}) {
        const tableBody = resolveElement(options.tableBody);
        const searchInput = resolveElement(options.searchInput);
        const gradeFilter = resolveElement(options.gradeFilter);
        const reloadButton = resolveElement(options.reloadButton);
        const notifications = resolveElement(options.notifications);
        const apiFetch = options.apiFetch;
        const apiPath = options.apiPath || '/students.php?action=list';
        const onViewLinks = typeof options.onViewLinks === 'function' ? options.onViewLinks : null;
        const tableColspan = Number(options.tableColspan) || 6;
        const formatCurrencyGs = options.formatCurrencyGs || defaultFormatCurrencyGs;
        const loadingMessage = options.loadingMessage || 'Cargando alumnos...';
        const emptyMessage = options.emptyMessage || 'No hay alumnos cargados';
        const filteredEmptyMessage = options.filteredEmptyMessage || 'No se encontraron alumnos con los filtros actuales';
        const errorMessage = options.errorMessage || 'No se pudieron cargar los alumnos.';

        const state = {
            students: [],
            filters: {
                query: '',
                grade: '',
            },
        };

        function showToast(message, type = 'info') {
            if (typeof options.notify === 'function') {
                options.notify(message, type);
                return;
            }
            createToast(notifications, message, type);
        }

        function renderRows(students = [], message = emptyMessage) {
            if (!tableBody) return;
            if (!students.length) {
                tableBody.innerHTML = `<tr><td colspan="${tableColspan}" class="muted">${escapeHtml(message)}</td></tr>`;
                return;
            }

            tableBody.innerHTML = '';
            students.forEach((student) => {
                const internalId = student.id;
                const externalId = student.external_id || null;
                const displayId = externalId || internalId;
                const copyValue = externalId || String(internalId);
                const linkedParentCount = Number(student.linked_parent_count) || 0;
                const isLinked = Boolean(student.is_linked);
                const linkedLabel = isLinked
                    ? `Vinculado${linkedParentCount > 1 ? ` (${linkedParentCount})` : ''}`
                    : 'Sin vínculo';

                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><code>${escapeHtml(displayId)}</code></td>
                    <td>${escapeHtml(student.name || '—')}</td>
                    <td>${escapeHtml(student.grade || '—')}</td>
                    <td><span class="status-pill ${isLinked ? 'linked' : 'unlinked'}">${escapeHtml(linkedLabel)}</span></td>
                    <td>${escapeHtml(formatCurrencyGs(student.balance))}</td>
                    <td class="actions">
                        <button class="secondary-btn" data-student-links="${escapeHtml(String(internalId))}">Ver vínculos</button>
                        <button class="ghost-btn" data-student-copy="${escapeHtml(copyValue)}">Copiar ID</button>
                    </td>
                `;
                tableBody.appendChild(row);
            });

            tableBody.querySelectorAll('button[data-student-links]').forEach((button) => {
                button.addEventListener('click', () => {
                    if (onViewLinks) {
                        onViewLinks(button.dataset.studentLinks);
                    }
                });
            });

            tableBody.querySelectorAll('button[data-student-copy]').forEach((button) => {
                button.addEventListener('click', () => {
                    const value = button.dataset.studentCopy;
                    navigator.clipboard?.writeText(value).then(() => {
                        showToast(`ID copiado: ${value}`, 'success');
                    }).catch(() => {
                        showToast('No se pudo copiar el ID', 'error');
                    });
                });
            });
        }

        function populateGradeFilter(students = []) {
            if (!gradeFilter) return;

            const grades = [...new Set(
                students
                    .map((student) => String(student.grade || '').trim())
                    .filter(Boolean)
            )].sort((a, b) => a.localeCompare(b, 'es', { numeric: true, sensitivity: 'base' }));

            const currentValue = gradeFilter.value || '';
            gradeFilter.innerHTML = '<option value="">Todos los grados</option>';

            grades.forEach((grade) => {
                const option = document.createElement('option');
                option.value = grade;
                option.textContent = grade;
                gradeFilter.appendChild(option);
            });

            if (currentValue && grades.includes(currentValue)) {
                gradeFilter.value = currentValue;
                state.filters.grade = currentValue;
                return;
            }

            gradeFilter.value = '';
            state.filters.grade = '';
        }

        function getFilteredStudents() {
            const query = normalizeFilterValue(state.filters.query);
            const selectedGrade = String(state.filters.grade || '').trim();

            return state.students.filter((student) => {
                const name = normalizeFilterValue(student.name);
                const internalId = normalizeFilterValue(student.id);
                const externalId = normalizeFilterValue(student.external_id);
                const grade = String(student.grade || '').trim();

                const matchesQuery = !query
                    || name.includes(query)
                    || internalId.includes(query)
                    || externalId.includes(query);

                const matchesGrade = !selectedGrade || grade === selectedGrade;

                return matchesQuery && matchesGrade;
            });
        }

        function applyFilters() {
            if (!tableBody) return;
            if (!state.students.length) {
                renderRows([], emptyMessage);
                return;
            }

            renderRows(getFilteredStudents(), filteredEmptyMessage);
        }

        async function load() {
            if (!tableBody || typeof apiFetch !== 'function') return [];
            try {
                tableBody.innerHTML = `<tr><td colspan="${tableColspan}" class="muted">${escapeHtml(loadingMessage)}</td></tr>`;
                const students = await apiFetch(apiPath);
                const studentsArray = Array.isArray(students) ? students : [];
                state.students = studentsArray;
                populateGradeFilter(studentsArray);
                applyFilters();
                return studentsArray;
            } catch (error) {
                showToast(`Error cargando alumnos: ${error.message}`, 'error');
                tableBody.innerHTML = `<tr><td colspan="${tableColspan}" class="muted">${escapeHtml(errorMessage)}</td></tr>`;
                return [];
            }
        }

        searchInput?.addEventListener('input', () => {
            state.filters.query = searchInput.value || '';
            applyFilters();
        });

        gradeFilter?.addEventListener('change', () => {
            state.filters.grade = gradeFilter.value || '';
            applyFilters();
        });

        reloadButton?.addEventListener('click', () => {
            load();
        });

        return {
            applyFilters,
            getFilteredStudents,
            load,
            renderRows,
            state,
        };
    }

    global.CantinaStudentsDirectory = {
        createStudentsDirectory,
    };
}(window));
