<?php

declare(strict_types=1);

/**
 * Cantina Face - Bootstrap Core (Minimal)
 * HostGator-ready: shared hosting, Apache + PHP 8.3
 * 
 * This bootstrap loads only essential components for health checks and simple endpoints.
 * NO session, NO auth, NO domain logic.
 */

// Prevent any output before we control it
ob_start();

$appDir = __DIR__;
$rootDir = dirname($appDir);

// Debug mode detection
$debugMode = isset($_GET['debug']) && $_GET['debug'] === '1';

// Load helpers first
require_once $appDir . '/helpers.php';

/**
 * Check if debug mode is enabled
 */
function is_debug_mode(): bool
{
    global $debugMode;
    return $debugMode;
}

/**
 * Send JSON error response and exit
 */
function bootstrap_core_json_error(string $message, int $code = 500): void
{
    ob_clean();
    http_response_code($code);
    header('Content-Type: application/json; charset=utf-8');
    
    $response = [
        'ok' => false,
        'error' => is_debug_mode() ? $message : 'Internal server error',
    ];
    
    if (is_debug_mode()) {
        $response['detail'] = $message;
        $response['last_error'] = error_get_last();
        $response['called_from'] = debug_backtrace(DEBUG_BACKTRACE_IGNORE_ARGS, 8);
        $response['script'] = $_SERVER['SCRIPT_FILENAME'] ?? 'unknown';
        $response['cwd'] = getcwd() ?: 'unknown';
    }
    
    echo json_encode($response, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

// Load configuration
$configPath = null;

if (is_file($appDir . '/config.php')) {
    $configPath = $appDir . '/config.php';
} elseif (is_file($appDir . '/config.example.php')) {
    $configPath = $appDir . '/config.example.php';
}

if ($configPath === null) {
    bootstrap_core_json_error('Config file missing. Create app/config.php from app/config.example.php.', 500);
}

$config = require $configPath;

if (!is_array($config)) {
    bootstrap_core_json_error('Config file must return an array.', 500);
}

// Store config in app registry
app_set('config', $config);

// Set timezone
date_default_timezone_set($config['app']['timezone'] ?? 'UTC');

// Load response helpers
require_once $appDir . '/response.php';

// Load logger if available (non-breaking)
$loggerPath = $appDir . '/logger.php';
if (is_file($loggerPath)) {
    require_once $loggerPath;
}

/**
 * Database helper - supports sqlite (default) and mysql
 * @return PDO
 */
function db(): PDO
{
    $pdo = app_get('db');
    if ($pdo instanceof PDO) {
        return $pdo;
    }

    $config = app_get('config', []);
    $dbConfig = $config['db'] ?? [];

    $driver = $dbConfig['driver'] ?? 'sqlite';

    try {
        if ($driver === 'sqlite') {
            $dbPath = $dbConfig['path'] ?? dirname(__DIR__) . '/data/db.sqlite';
            $dbDir = dirname($dbPath);

            if (!is_dir($dbDir)) {
                if (!mkdir($dbDir, 0775, true) && !is_dir($dbDir)) {
                    throw new RuntimeException('Failed to create SQLite directory: ' . $dbDir);
                }
            }

            $dsn = 'sqlite:' . $dbPath;
            $options = [
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_EMULATE_PREPARES => false,
            ];
            $pdo = new PDO($dsn, null, null, $options);

            require_once __DIR__ . '/db_schema.php';

            // ✅ AUTO INIT SCHEMA si falta
            ensure_sqlite_schema($pdo, $dbPath);
        } else {
            // MySQL
            $dsn = sprintf(
                'mysql:host=%s;port=%d;dbname=%s;charset=%s',
                $dbConfig['host'] ?? 'localhost',
                $dbConfig['port'] ?? 3306,
                $dbConfig['database'] ?? '',
                $dbConfig['charset'] ?? 'utf8mb4'
            );
            $options = [
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_EMULATE_PREPARES => false,
            ];
            $pdo = new PDO(
                $dsn,
                $dbConfig['username'] ?? '',
                $dbConfig['password'] ?? '',
                $options
            );
        }
    } catch (PDOException $e) {
        // Log error but don't expose details
        if (function_exists('log_error')) {
            log_error('Database connection failed', ['driver' => $driver]);
        }
        bootstrap_core_json_error('Database connection failed', 500);
    }

    app_set('db', $pdo);
    return $pdo;
}

// Error handlers - must output JSON only
set_error_handler(function ($severity, $message, $file, $line) {
    $errorTypes = [
        E_ERROR => 'Error',
        E_WARNING => 'Warning',
        E_PARSE => 'Parse',
        E_NOTICE => 'Notice',
        E_CORE_ERROR => 'Core Error',
        E_CORE_WARNING => 'Core Warning',
        E_COMPILE_ERROR => 'Compile Error',
        E_COMPILE_WARNING => 'Compile Warning',
        E_USER_ERROR => 'User Error',
        E_USER_WARNING => 'User Warning',
        E_USER_NOTICE => 'User Notice',
    ];

    $type = $errorTypes[$severity] ?? 'Unknown';

    if (function_exists('log_error')) {
        log_error('PHP error', [
            'type' => $type,
            'message' => $message,
            'file' => $file,
            'line' => $line,
        ]);
    }

    // Fatal errors should return JSON
    if (in_array($severity, [E_ERROR, E_CORE_ERROR, E_COMPILE_ERROR, E_USER_ERROR], true)) {
        if (is_debug_mode()) {
            bootstrap_core_json_error("[$type] $message at $file:$line", 500);
        } else {
            bootstrap_core_json_error('Internal server error', 500);
        }
    }

    return true; // Suppress default error handling
});

set_exception_handler(function (Throwable $exception) {
    if (function_exists('log_error')) {
        log_error('Uncaught exception', [
            'type' => get_class($exception),
            'message' => $exception->getMessage(),
            'file' => $exception->getFile(),
            'line' => $exception->getLine(),
        ]);
    }
    
    if (is_debug_mode()) {
        $message = $exception->getMessage();
        $file = $exception->getFile();
        $line = $exception->getLine();
        bootstrap_core_json_error("Exception: $message at $file:$line", 500);
    } else {
        bootstrap_core_json_error('Internal server error', 500);
    }
});

register_shutdown_function(function () {
    $error = error_get_last();
    if ($error && in_array($error['type'], [E_ERROR, E_CORE_ERROR, E_COMPILE_ERROR], true)) {
        if (function_exists('log_error')) {
            log_error('Fatal error', $error);
        }
        
        if (is_debug_mode()) {
            $message = $error['message'] ?? 'Unknown fatal error';
            $file = $error['file'] ?? 'unknown';
            $line = $error['line'] ?? 0;
            bootstrap_core_json_error("Fatal: $message at $file:$line", 500);
        } else {
            bootstrap_core_json_error('Internal server error', 500);
        }
    }
});

// Clean output buffer - we're ready
ob_end_clean();
