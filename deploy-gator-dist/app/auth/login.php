<?php

declare(strict_types=1);

require_once __DIR__ . '/../helpers.php';
require_once __DIR__ . '/../response.php';
require_once __DIR__ . '/../session.php';

const ROLE_ADMIN = 'admin';
const ROLE_PARENT = 'parent';
const ROLE_CAJERA = 'cajera';
const ROLE_STOCK = 'stock';

function hash_password(string $password): string
{
    return password_hash($password, PASSWORD_DEFAULT);
}

function verify_password_hash(string $password, string $hash): bool
{
    if ($hash === '' || $hash === null) {
        return false;
    }
    return password_verify($password, $hash);
}

function users_table(): string
{
    return 'users';
}

function find_user_by_email(string $email): ?array
{
    $pdo = db();
    $stmt = $pdo->prepare('SELECT * FROM ' . users_table() . ' WHERE email = :email LIMIT 1');
    $stmt->execute(['email' => strtolower($email)]);
    $row = $stmt->fetch(PDO::FETCH_ASSOC);
    return $row ?: null;
}

function find_user_by_id(int $id): ?array
{
    $pdo = db();
    $stmt = $pdo->prepare('SELECT * FROM ' . users_table() . ' WHERE id = :id LIMIT 1');
    $stmt->execute(['id' => $id]);
    $row = $stmt->fetch(PDO::FETCH_ASSOC);
    return $row ?: null;
}

function serialize_user(array $user): array
{
    return [
        'id' => (int) $user['id'],
        'email' => $user['email'],
        'full_name' => $user['full_name'] ?? '',
        'first_name' => $user['first_name'] ?? null,
        'last_name' => $user['last_name'] ?? null,
        'dni' => $user['dni'] ?? null,
        'phone' => $user['phone'] ?? null,
        'role' => $user['role'],
        'point_of_sale_id' => isset($user['point_of_sale_id']) ? (int) $user['point_of_sale_id'] : null,
        'is_active' => (bool) ($user['is_active'] ?? 1),
        'created_at' => $user['created_at'] ?? null,
        'updated_at' => $user['updated_at'] ?? null,
    ];
}

function ensure_user_active(array $user): void
{
    if (isset($user['is_active']) && !(int) $user['is_active']) {
        json_error('Usuario deshabilitado', 403);
    }
}

function authenticate_credentials(string $email, string $password): array
{
    $user = find_user_by_email($email);
    if (!$user || !verify_password_hash($password, $user['password_hash'] ?? '')) {
        json_error('Credenciales inválidas', 401);
    }
    ensure_user_active($user);
    return $user;
}

function issue_session_for_user(array $user): array
{
    session_regenerate_id(true);
    session_set_user($user);
    $_SESSION['csrf_token'] = bin2hex(random_bytes(16));

    return [
        'token' => session_id(),
        'csrf_token' => $_SESSION['csrf_token'],
        'user' => serialize_user($user),
    ];
}

function require_csrf_token(): void
{
    if (http_method() === 'GET') {
        return;
    }

    $header = $_SERVER['HTTP_X_CSRF_TOKEN'] ?? '';
    $token = $_POST['_token'] ?? $header;
    $expected = $_SESSION['csrf_token'] ?? '';
    if (!$expected || !hash_equals($expected, (string) $token)) {
        json_error('CSRF token inválido', 419);
    }
}

