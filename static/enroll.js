// Global variables for enrollment
let enrollVideoStream = null;
let currentPreview = null;
let enrollmentState = 'ready';
let enrollExistingStudent = null;

// DOM elements
const enrollVideo = document.getElementById('enroll-video');
const studentNameInput = document.getElementById('student-name');
const studentGradeInput = document.getElementById('student-grade');
const studentBalanceInput = document.getElementById('student-balance');
const previewImage = document.getElementById('preview-image');
const noPreview = document.getElementById('no-preview');
const embeddingState = document.getElementById('embedding-state');
const enrollSubmit = document.getElementById('enroll-submit');
const enrollCancel = document.getElementById('enroll-cancel');
const captureOneShot = document.getElementById('capture-one-shot');
const captureBurst = document.getElementById('capture-burst');
const enrollExistingSearchInput = document.getElementById('enroll-existing-search');
const enrollExistingResults = document.getElementById('enroll-existing-results');
const enrollExistingSelected = document.getElementById('enroll-existing-selected');

// Initialize enrollment page
async function initEnrollment() {
    try {
        await startEnrollCamera();
        setupEnrollEventListeners();
        updateEnrollmentState('ready');
        showNotification('Cámara lista para registro', 'success');
    } catch (error) {
        showNotification('Error al inicializar cámara: ' + error.message, 'error');
        updateEnrollmentState('error');
    }

    // Existing student search
    if (enrollExistingSearchInput) {
        let searchTimer = null;
        enrollExistingSearchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimer);
            const q = e.target.value.trim();
            if (!q) {
                enrollExistingResults.innerHTML = '';
                enrollExistingSelected.textContent = '';
                enrollExistingStudent = null;
                validateForm();
                return;
            }
            searchTimer = setTimeout(() => searchExistingStudents(q), 300);
        });
    }
}

// Start enrollment camera
async function startEnrollCamera() {
    try {
        enrollVideoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });

        enrollVideo.srcObject = enrollVideoStream;
        document.getElementById('camera-indicator').textContent = '✅';
        document.getElementById('camera-indicator').className = 'status-indicator success';
    } catch (error) {
        document.getElementById('camera-indicator').textContent = '❌';
        document.getElementById('camera-indicator').className = 'status-indicator error';
        throw new Error('No se pudo acceder a la cámara');
    }
}

// Setup event listeners for enrollment
function setupEnrollEventListeners() {
    // Form inputs
    studentNameInput.addEventListener('input', validateForm);
    studentGradeInput.addEventListener('input', validateForm);
    studentBalanceInput.addEventListener('input', validateForm);

    // Capture buttons
    captureOneShot.addEventListener('click', () => capturePhoto('one-shot'));
    captureBurst.addEventListener('click', () => capturePhoto('burst'));

    // Action buttons
    enrollSubmit.addEventListener('click', submitEnrollment);
    enrollCancel.addEventListener('click', () => {
        window.location.href = '/index.html';
    });

    // Back button
    const backButton = document.getElementById('back-to-main');
    if (backButton) {
        backButton.addEventListener('click', () => {
            window.location.href = '/index.html';
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Space for capture
        if (e.key === ' ' && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            capturePhoto('one-shot');
            return;
        }

        // Shift+Space for burst
        if (e.key === ' ' && (e.shiftKey)) {
            e.preventDefault();
            capturePhoto('burst');
            return;
        }

        // Ctrl+S for submit
        if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            if (!enrollSubmit.disabled) {
                submitEnrollment();
            }
            return;
        }

        // Ctrl+Left Arrow for cancel
        if (e.key === 'ArrowLeft' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            window.location.href = '/index.html';
            return;
        }

        // Escape to cancel
        if (e.key === 'Escape') {
            e.preventDefault();
            window.location.href = '/index.html';
            return;
        }
    });

    // Focus first input
    studentNameInput.focus();
}

// Capture photo (one-shot or burst)
async function capturePhoto(mode) {
    if (enrollmentState === 'processing') return;

    updateEnrollmentState('processing');

    try {
        if (mode === 'one-shot') {
            await captureSingleFrame();
        } else if (mode === 'burst') {
            await captureBurstFrames();
        }

        updateEnrollmentState('ready');
        validateForm();

    } catch (error) {
        showNotification('Error en captura: ' + error.message, 'error');
        updateEnrollmentState('error');
    }
}

// Capture single frame
async function captureSingleFrame() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = enrollVideo.videoWidth;
    canvas.height = enrollVideo.videoHeight;
    ctx.drawImage(enrollVideo, 0, 0);

    return new Promise((resolve) => {
        canvas.toBlob(async (blob) => {
            currentPreview = blob;
            updatePreview(blob);
            resolve();
        }, 'image/jpeg', 0.9);
    });
}

// Capture burst of frames
async function captureBurstFrames() {
    const frames = [];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = enrollVideo.videoWidth;
    canvas.height = enrollVideo.videoHeight;

    const isAttachMode = !!enrollExistingStudent;

    // No usar ráfaga para modo "asignar a alumno existente" para evitar inconsistencias
    if (isAttachMode) {
        showNotification('Para asignar la cara a un alumno existente usa la captura rápida (1 foto).', 'error');
        return;
    }

    const name = studentNameInput.value.trim();
    const grade = studentGradeInput.value.trim();
    if (!name || !grade) {
        showNotification('Complete nombre y grado antes de usar la captura en ráfaga', 'error');
        return;
    }

    // Capture 5 frames over 2 seconds
    for (let i = 0; i < 5; i++) {
        ctx.drawImage(enrollVideo, 0, 0);
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.9);
        });
        frames.push(blob);

        if (i < 4) { // Don't wait after last frame
            await new Promise(resolve => setTimeout(resolve, 400)); // 400ms between frames
        }
    }

    // Create FormData with multiple frames (solo para nuevo alumno, con datos completos)
    const formData = new FormData();
    formData.append('name', name);
    formData.append('grade', grade);
    frames.forEach((frame, index) => {
        formData.append('frames', frame, `frame_${index}.jpg`);
    });

    // Call burst enrollment API
    const response = await fetch('/api/enroll_burst', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        let detail = 'Error en registro por ráfaga';
        try {
            const errJson = await response.json();
            detail = errJson.detail || detail;
        } catch (_) {
            try { detail = await response.text(); } catch (_) {}
        }
        console.error('[enroll_burst] status:', response.status, 'detail:', detail);
        throw new Error(detail);
    }

    const result = await response.json();

    // Show success with captured image
    if (result.photo_path) {
        currentPreview = frames[0]; // Use first frame as preview
        updatePreview(frames[0]);
        embeddingState.textContent = '✅ Calculado';
        showNotification(`✅ Registrado: ${result.name}`, 'success');
    }
}

// Update preview image
function updatePreview(blob) {
    if (blob) {
        const url = URL.createObjectURL(blob);
        previewImage.src = url;
        previewImage.classList.remove('hidden');
        noPreview.classList.add('hidden');
    }
}

// Update enrollment state
function updateEnrollmentState(state) {
    enrollmentState = state;
    document.getElementById('enrollment-state').textContent = getStateText(state);

    // Update UI based on state
    const isReady = state === 'ready';
    captureOneShot.disabled = !isReady;
    captureBurst.disabled = !isReady;
}

// Get state text
function getStateText(state) {
    switch (state) {
        case 'ready': return 'Listo';
        case 'processing': return 'Procesando...';
        case 'error': return 'Error';
        default: return 'Listo';
    }
}

// Validate form and update submit button
function validateForm() {
    const isAttachMode = !!enrollExistingStudent;
    const hasData = isAttachMode || (studentNameInput.value.trim() && studentGradeInput.value.trim());
    const hasPreview = currentPreview !== null;
    enrollSubmit.disabled = !(hasData && hasPreview);
}

// Submit enrollment
async function submitEnrollment() {
    if (!currentPreview || enrollmentState === 'processing') return;

    const name = studentNameInput.value.trim();
    const grade = studentGradeInput.value.trim();
    const balance = parseInt(studentBalanceInput.value) || 0;

    const isAttachMode = !!enrollExistingStudent;

    if (!isAttachMode && (!name || !grade)) {
        showNotification('Complete nombre y grado', 'error');
        return;
    }

    updateEnrollmentState('processing');

    try {
        if (isAttachMode) {
            // Attach current preview to existing student
            const formData = new FormData();
            formData.append('frame', currentPreview, 'photo.jpg');

            const response = await fetch(`/api/students/${enrollExistingStudent.id}/attach-face`, {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                let detail = result.detail || 'Error al asociar cara al alumno';
                throw new Error(detail);
            }

            embeddingState.textContent = '✅ Calculado';
            showNotification(`✅ Cara asociada a ${enrollExistingStudent.name}`, 'success');

            // Reset solo selección existente y preview
            resetForm();
        } else {
            const formData = new FormData();
            formData.append('name', name);
            formData.append('grade', grade);
            formData.append('frame', currentPreview, 'photo.jpg');

            const response = await fetch('/api/enroll_one_shot', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let detail = 'Error en registro';
                try {
                    const errJson = await response.json();
                    detail = errJson.detail || detail;
                } catch (_) {
                    try { detail = await response.text(); } catch (_) {}
                }
                console.error('[enroll_one_shot] status:', response.status, 'detail:', detail);
                throw new Error(detail);
            }

            const result = await response.json();

            embeddingState.textContent = result.embedding_computed ? '✅ Calculado' : '❌ No calculado';
            showNotification(`✅ Registrado: ${result.name}`, 'success');

            // Reset form for next enrollment
            resetForm();
        }

    } catch (error) {
        showNotification('Error: ' + error.message, 'error');
        embeddingState.textContent = '❌ Error';
    } finally {
        updateEnrollmentState('ready');
    }
}

// Reset form
function resetForm() {
    studentNameInput.value = '';
    studentGradeInput.value = '';
    studentBalanceInput.value = '0';
    currentPreview = null;
    previewImage.classList.add('hidden');
    noPreview.classList.remove('hidden');
    embeddingState.textContent = 'No calculado';
    enrollSubmit.disabled = true;
    enrollExistingStudent = null;
    if (enrollExistingSearchInput) enrollExistingSearchInput.value = '';
    if (enrollExistingResults) enrollExistingResults.innerHTML = '';
    if (enrollExistingSelected) enrollExistingSelected.textContent = '';
    validateForm();
}

// Buscar alumnos existentes para adjuntar cara
async function searchExistingStudents(query) {
    const trimmed = (query || '').trim();
    if (!trimmed) {
        enrollExistingResults.innerHTML = '';
        enrollExistingSelected.textContent = '';
        enrollExistingStudent = null;
        validateForm();
        return;
    }

    try {
        const response = await fetch(`/api/students?query=${encodeURIComponent(trimmed)}`);
        let students = await response.json();

        // Filtrar registros de prueba con nombre 'temp'
        students = (students || []).filter(s => (s.name || '').toLowerCase() !== 'temp');

        enrollExistingResults.innerHTML = '';
        if (!students || students.length === 0) {
            enrollExistingResults.innerHTML = '<p class="no-data">Sin resultados</p>';
            enrollExistingStudent = null;
            enrollExistingSelected.textContent = '';
            validateForm();
            return;
        }

        students.forEach((student) => {
            const div = document.createElement('div');
            div.className = 'search-result';
            div.innerHTML = `
                <div class="search-info">
                    <div class="search-name">${student.name}</div>
                    <div class="search-grade">${student.grade}</div>
                </div>
            `;

            div.addEventListener('click', () => {
                enrollExistingStudent = student;
                enrollExistingSelected.textContent = `Asignando cara a: ${student.name} (${student.grade})`;
                enrollExistingResults.innerHTML = '';
                validateForm();
            });

            enrollExistingResults.appendChild(div);
        });
    } catch (error) {
        console.error('Error buscando alumnos existentes:', error);
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

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (enrollVideoStream) {
        enrollVideoStream.getTracks().forEach(track => track.stop());
    }
});

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initEnrollment);
