<?php

declare(strict_types=1);

require_once __DIR__ . '/../session.php';
require_once __DIR__ . '/../response.php';
require_once __DIR__ . '/../logger.php';

function current_user(): ?array
{
    return session_user();
}

function require_auth(array $roles = []): array
{
    $user = current_user();
    if (!$user) {
        json_error('No autenticado', 401);
    }
    if ($roles && !in_array($user['role'], $roles, true)) {
        json_error('Permisos insuficientes', 403);
    }
    return $user;
}

function rate_limit_check(string $key, int $maxAttempts = 60, int $windowSeconds = 60): void
{
    $storePath = config('app.rate_limiter_path', base_path('storage/logs/ratelimit.json'));
    $bucket = [];
    if (is_file($storePath)) {
        $bucket = json_decode((string) file_get_contents($storePath), true) ?: [];
    }
    $now = time();
    $entry = $bucket[$key] ?? ['count' => 0, 'start' => $now];
    if (($now - $entry['start']) > $windowSeconds) {
        $entry = ['count' => 0, 'start' => $now];
    }
    $entry['count']++;
    $bucket[$key] = $entry;
    if ($entry['count'] > $maxAttempts) {
        json_error('Demasiadas solicitudes', 429);
    }
    $dir = dirname($storePath);
    if (!is_dir($dir)) {
        mkdir($dir, 0775, true);
    }
    file_put_contents($storePath, json_encode($bucket));
}

