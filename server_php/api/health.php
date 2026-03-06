<?php

declare(strict_types=1);

require_once __DIR__ . '/../app/bootstrap_core.php';

/**
 * Health check endpoint for HostGator deployment
 * Returns system status and connectivity information
 */

// Check for debug mode
$debugMode = isset($_GET['debug']) && $_GET['debug'] === '1';

$response = [
    'ok' => true,
    'php' => PHP_VERSION,
    'config_loaded' => false,
    'db_ok' => false,
    'driver' => null,
    'sqlite_path' => null,
    'sqlite_writable' => null,
    'error' => null,
];

$statusCode = 200;

// Check config loaded
$config = app_get('config', []);
if (!empty($config) && is_array($config)) {
    $response['config_loaded'] = true;

    // Get database driver info
    $dbConfig = $config['db'] ?? [];
    $driver = $dbConfig['driver'] ?? 'sqlite';
    $response['driver'] = $driver;

    if ($driver === 'sqlite') {
        $dbPath = $dbConfig['path'] ?? dirname(__DIR__) . '/data/db.sqlite';
        $response['sqlite_path'] = $dbPath;
        $response['sqlite_writable'] = is_writable(dirname($dbPath)) || (file_exists($dbPath) && is_writable($dbPath));
    }
} else {
    $response['ok'] = false;
    $response['error'] = 'Configuration not loaded';
    $statusCode = 500;
}

// Check database connectivity if config is loaded
if ($response['config_loaded']) {
    try {
        $pdo = db();
        // Try a simple query
        $pdo->query('SELECT 1');
        $response['db_ok'] = true;
    } catch (Throwable $e) {
        $response['db_ok'] = false;
        $response['ok'] = false;

        // Only show detailed error in debug mode
        if ($debugMode) {
            $response['error'] = 'Database connection failed: ' . $e->getMessage();
            $response['error_class'] = get_class($e);
        } else {
            $response['error'] = 'Database connection failed';
        }

        $statusCode = 500;
    }
}

// Return response
http_response_code($statusCode);
header('Content-Type: application/json; charset=utf-8');
echo json_encode($response, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
