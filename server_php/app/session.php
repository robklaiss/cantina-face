<?php

declare(strict_types=1);

require_once __DIR__ . '/helpers.php';

function session_bootstrap(): void
{
    $config = config('session', []);
    $name = $config['name'] ?? 'CANTINASESSID';
    $lifetime = (int) ($config['lifetime'] ?? 3600);

    session_name($name);
    session_set_cookie_params([
        'lifetime' => $lifetime,
        'path' => '/',
        'domain' => '',
        'secure' => isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off',
        'httponly' => true,
        'samesite' => 'Lax',
    ]);

    if (session_status() !== PHP_SESSION_ACTIVE) {
        session_start();
    }

    if (!isset($_SESSION['initiated'])) {
        session_regenerate_id(true);
        $_SESSION['initiated'] = true;
        $_SESSION['created_at'] = time();
    }

    $lastActivity = $_SESSION['last_activity'] ?? 0;
    if ($lastActivity && (time() - $lastActivity) > $lifetime) {
        session_destroy();
        session_start();
    }
    $_SESSION['last_activity'] = time();
}

function session_user(): ?array
{
    return $_SESSION['user'] ?? null;
}

function session_set_user(array $user): void
{
    $_SESSION['user'] = [
        'id' => (int) $user['id'],
        'email' => $user['email'],
        'role' => $user['role'],
        'full_name' => $user['full_name'] ?? '',
        'first_name' => $user['first_name'] ?? null,
        'last_name' => $user['last_name'] ?? null,
    ];
}

function session_logout(): void
{
    $_SESSION = [];
    if (ini_get('session.use_cookies')) {
        $params = session_get_cookie_params();
        setcookie(session_name(), '', time() - 42000, $params['path'], $params['domain'], $params['secure'], $params['httponly']);
    }
    session_destroy();
}

