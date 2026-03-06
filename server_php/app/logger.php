<?php

declare(strict_types=1);

require_once __DIR__ . '/helpers.php';

function app_log(string $level, string $message, array $context = []): void
{
    $logPath = config('app.log_path');
    if (!$logPath) {
        return;
    }

    $dir = dirname($logPath);
    if (!is_dir($dir) && !mkdir($dir, 0775, true) && !is_dir($dir)) {
        return;
    }

    $line = sprintf(
        "[%s] %s.%s [%s] %s %s\n",
        now(),
        php_sapi_name(),
        $_SERVER['REMOTE_ADDR'] ?? 'cli',
        strtoupper($level),
        $message,
        $context ? json_encode($context, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) : ''
    );

    file_put_contents($logPath, $line, FILE_APPEND);
}

function log_info(string $message, array $context = []): void
{
    app_log('info', $message, $context);
}

function log_warning(string $message, array $context = []): void
{
    app_log('warning', $message, $context);
}

function log_error(string $message, array $context = []): void
{
    app_log('error', $message, $context);
}

