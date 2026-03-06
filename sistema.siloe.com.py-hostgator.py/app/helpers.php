<?php

declare(strict_types=1);

function app_store(?string $key = null, $value = null)
{
    static $store = [];
    if ($key === null) {
        return $store;
    }

    if (func_num_args() === 2) {
        $store[$key] = $value;
    }

    return $store[$key] ?? null;
}

function app_get(string $key, $default = null)
{
    return app_store($key) ?? $default;
}

function app_set(string $key, $value): void
{
    app_store($key, $value);
}

function base_path(string $path = ''): string
{
    $root = dirname(__DIR__);
    return rtrim($root . '/' . ltrim($path, '/'), '/');
}

function config(string $key, $default = null)
{
    $config = app_get('config', []);
    if (!$key) {
        return $config;
    }

    $segments = explode('.', $key);
    $value = $config;
    foreach ($segments as $segment) {
        if (!is_array($value) || !array_key_exists($segment, $value)) {
            return $default;
        }
        $value = $value[$segment];
    }

    return $value;
}

function env_bool($value, bool $default = false): bool
{
    if ($value === null) {
        return $default;
    }
    if (is_bool($value)) {
        return $value;
    }
    return in_array(strtolower((string) $value), ['1', 'true', 'yes', 'on'], true);
}

function request_json(): array
{
    $raw = file_get_contents('php://input');
    if ($raw === false || $raw === '') {
        return [];
    }

    $data = json_decode($raw, true);
    return is_array($data) ? $data : [];
}

function request_body(): array
{
    if ($_SERVER['CONTENT_TYPE'] ?? '' === 'application/json') {
        return request_json();
    }
    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
        return $_POST;
    }
    return [];
}

function sanitize_string(?string $value): ?string
{
    if ($value === null) {
        return null;
    }
    return trim(filter_var($value, FILTER_UNSAFE_RAW, FILTER_FLAG_STRIP_LOW));
}

function now(): string
{
    return (new DateTimeImmutable('now', new DateTimeZone(config('app.timezone', 'UTC'))))->format('Y-m-d H:i:s');
}

function parse_int($value, ?int $default = null): ?int
{
    if ($value === null || $value === '') {
        return $default;
    }
    return filter_var($value, FILTER_VALIDATE_INT) !== false ? (int) $value : $default;
}

function parse_float($value, ?float $default = null): ?float
{
    if ($value === null || $value === '') {
        return $default;
    }
    return filter_var($value, FILTER_VALIDATE_FLOAT) !== false ? (float) $value : $default;
}

function input(string $key, $default = null)
{
    return $_REQUEST[$key] ?? $default;
}

function http_method(): string
{
    return strtoupper($_SERVER['REQUEST_METHOD'] ?? 'GET');
}

function get_client_ip(): string
{
    return $_SERVER['HTTP_X_FORWARDED_FOR']
        ?? $_SERVER['HTTP_CLIENT_IP']
        ?? $_SERVER['REMOTE_ADDR']
        ?? '0.0.0.0';
}

