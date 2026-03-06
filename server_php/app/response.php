<?php

declare(strict_types=1);

function json_response(array $payload, int $status = 200): void
{
    http_response_code($status);
    header('Content-Type: application/json');
    echo json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

function json_ok($data = null, int $status = 200): void
{
    $payload = ['ok' => true];
    if ($data !== null) {
        $payload['data'] = $data;
    }
    json_response($payload, $status);
}

function json_error(string $message, int $status = 400, array $extra = []): void
{
    $payload = array_merge(['ok' => false, 'error' => $message], $extra);
    json_response($payload, $status);
}

