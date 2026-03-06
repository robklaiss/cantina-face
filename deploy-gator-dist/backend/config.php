<?php
// Basic configuration for Deploy-Gator backend

define('DEPLOY_TOKEN', 'CHANGE_ME_DEPLOY_TOKEN');
define('RELEASES_DIR', __DIR__ . '/releases');
define('TMP_DIR', __DIR__ . '/tmp');
define('LATEST_JSON_PATH', __DIR__ . '/latest.json');

date_default_timezone_set('UTC');

if (!is_dir(RELEASES_DIR) && !mkdir(RELEASES_DIR, 0775, true)) {
    http_response_code(500);
    die('Failed to ensure releases directory');
}

if (!is_dir(TMP_DIR) && !mkdir(TMP_DIR, 0775, true)) {
    http_response_code(500);
    die('Failed to ensure tmp directory');
}

function respond_json(array $payload, int $status = 200): void
{
    http_response_code($status);
    header('Content-Type: application/json');
    echo json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    exit;
}

function verify_token(?string $token): void
{
    if (!$token || !hash_equals(DEPLOY_TOKEN, $token)) {
        respond_json([
            'ok' => false,
            'error' => 'Unauthorized',
        ], 403);
    }
}

function release_version_from_filename(string $filename): string
{
    if (preg_match('/project_(\d{8}_\d{6})\.zip$/', $filename, $m)) {
        return $m[1];
    }

    return pathinfo($filename, PATHINFO_FILENAME) ?: $filename;
}

function release_metadata(string $filename): ?array
{
    $filepath = RELEASES_DIR . '/' . $filename;

    if (!is_file($filepath)) {
        return null;
    }

    return [
        'version' => release_version_from_filename($filename),
        'filename' => $filename,
        'sha256' => hash_file('sha256', $filepath),
        'bytes' => filesize($filepath),
        'uploaded_at' => gmdate('c', filemtime($filepath)),
    ];
}

function ensure_latest_structure(array $data = []): array
{
    $defaults = [
        'version' => null,
        'filename' => null,
        'sha256' => null,
        'bytes' => null,
        'uploaded_at' => null,
    ];

    return array_merge($defaults, array_intersect_key($data, $defaults));
}

function read_latest(): array
{
    if (!file_exists(LATEST_JSON_PATH)) {
        return ensure_latest_structure();
    }

    $decoded = json_decode((string) file_get_contents(LATEST_JSON_PATH), true);

    if (!is_array($decoded)) {
        return ensure_latest_structure();
    }

    return ensure_latest_structure($decoded);
}

function write_latest(array $data): void
{
    $payload = ensure_latest_structure($data);
    file_put_contents(LATEST_JSON_PATH, json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . "\n", LOCK_EX);
}
