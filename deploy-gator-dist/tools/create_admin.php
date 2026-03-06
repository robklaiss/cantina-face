<?php

declare(strict_types=1);

require_once __DIR__ . '/../app/bootstrap.php';

if (php_sapi_name() !== 'cli') {
    fwrite(STDERR, "Este script debe ejecutarse desde CLI.\n");
    exit(1);
}

$options = getopt('', ['email:', 'password:', 'role::', 'name::']);

$email = $options['email'] ?? null;
$password = $options['password'] ?? null;
$role = $options['role'] ?? ROLE_ADMIN;
$name = $options['name'] ?? 'Administrador';

if (!$email || !$password) {
    fwrite(STDERR, "Uso: php tools/create_admin.php --email=EMAIL --password=PASSWORD [--role=admin|cajera|stock] [--name=Nombre]\n");
    exit(1);
}

try {
    $payload = [
        'email' => $email,
        'password' => $password,
        'role' => $role,
        'full_name' => $name,
    ];
    $user = create_user_admin($payload);
    fwrite(STDOUT, "Usuario creado/actualizado:\n");
    fwrite(STDOUT, json_encode($user, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE) . "\n");
    exit(0);
} catch (Throwable $e) {
    fwrite(STDERR, "Error: " . $e->getMessage() . "\n");
    exit(1);
}

