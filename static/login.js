const AUTH_TOKEN_KEY = 'cantina_face_token';
const AUTH_USER_KEY = 'cantina_face_user';
const AUTH_PROFILE_KEY = 'cantina_face_user_profile';

const loginForm = document.getElementById('login-form');
const loginError = document.getElementById('login-error');
const loginButton = loginForm?.querySelector('.login-btn');

const nativeFetch = window.fetch.bind(window);

function getNextUrl() {
    const params = new URLSearchParams(window.location.search);
    const next = params.get('next');
    if (!next || !next.startsWith('/')) {
        return '/index.html';
    }
    return next;
}

function setAuthSession(token, email) {
    if (token) {
        localStorage.setItem(AUTH_TOKEN_KEY, token);
        if (email) {
            localStorage.setItem(AUTH_USER_KEY, email);
        }
    } else {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
        localStorage.removeItem(AUTH_PROFILE_KEY);
    }
}

function setUserProfile(profile) {
    if (profile) {
        localStorage.setItem(AUTH_PROFILE_KEY, JSON.stringify(profile));
    } else {
        localStorage.removeItem(AUTH_PROFILE_KEY);
    }
}

async function fetchCurrentUser(token) {
    try {
        const response = await nativeFetch('/auth/me', {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) {
            throw new Error('No se pudo obtener el usuario actual');
        }
        const profile = await response.json();
        setUserProfile(profile);
        return profile;
    } catch (error) {
        console.warn('[login] No se pudo cargar el perfil:', error);
        setUserProfile(null);
        return null;
    }
}

async function verifyExistingSession() {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    if (!token) return false;

    try {
        const response = await nativeFetch('/auth/me', {
            headers: {
                'Authorization': `Bearer ${token}`,
            },
        });
        if (!response.ok) {
            throw new Error('Sesión inválida');
        }
        const profile = await response.json();
        setUserProfile(profile);
        redirectToApp();
        return true;
    } catch (error) {
        console.warn('[login] Sesión inválida, limpiando token', error);
        setAuthSession(null, null);
        return false;
    }
}

function redirectToApp() {
    window.location.href = getNextUrl();
}

function setLoading(isLoading) {
    if (!loginButton) return;
    loginButton.disabled = isLoading;
    loginButton.textContent = isLoading ? 'Ingresando…' : 'Iniciar sesión';
}

async function handleLoginSubmit(event) {
    event.preventDefault();
    if (!loginForm) return;

    const email = loginForm.elements.username.value.trim();
    const password = loginForm.elements.password.value.trim();

    if (!email || !password) {
        loginError.textContent = 'Completa tus credenciales';
        return;
    }

    loginError.textContent = '';
    setLoading(true);

    try {
        const payload = new URLSearchParams();
        payload.append('username', email);
        payload.append('password', password);

        const response = await nativeFetch('/auth/token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: payload,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || 'Credenciales incorrectas');
        }

        const data = await response.json();
        if (!data?.access_token) {
            throw new Error('Respuesta inválida del servidor');
        }

        setAuthSession(data.access_token, email);
        await fetchCurrentUser(data.access_token);
        redirectToApp();
    } catch (error) {
        loginError.textContent = error.message || 'No se pudo iniciar sesión';
    } finally {
        setLoading(false);
    }
}

async function bootstrapLogin() {
    await verifyExistingSession();
    loginForm?.addEventListener('submit', handleLoginSubmit);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrapLogin);
} else {
    bootstrapLogin();
}
