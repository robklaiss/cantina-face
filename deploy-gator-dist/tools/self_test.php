<?php

declare(strict_types=1);

require_once __DIR__ . '/../app/bootstrap.php';

if (php_sapi_name() !== 'cli') {
    fwrite(STDERR, "Este script debe ejecutarse desde CLI.\n");
    exit(1);
}

echo "Cantina Face - Self Test" . PHP_EOL;

try {
    // DB test
    $db = db();
    $db->query('SELECT 1');
    echo "[OK] Conexión a MySQL" . PHP_EOL;

    // Pending migrations (simple check: if any tables missing)
    $tables = ['users', 'students', 'transactions'];
    foreach ($tables as $table) {
        $stmt = $db->prepare('SHOW TABLES LIKE :table');
        $stmt->execute(['table' => $table]);
        if (!$stmt->fetch()) {
            echo "[WARN] Tabla faltante: {$table}\n";
        }
    }

    // Sample query counts
    $counts = db_fetch_all('SELECT \'users\' AS name, COUNT(*) AS total FROM users
        UNION ALL
        SELECT \'students\' AS name, COUNT(*) FROM students
        UNION ALL
        SELECT \'transactions\' AS name, COUNT(*) FROM transactions');
    foreach ($counts as $row) {
        echo sprintf("[DATA] %s: %d\n", ucfirst($row['name']), (int) $row['total']);
    }

    // Folder permissions check
    $paths = [__DIR__ . '/../storage', __DIR__ . '/../storage/logs', __DIR__ . '/../storage/cache'];
    foreach ($paths as $path) {
        if (!is_dir($path)) {
            mkdir($path, 0775, true);
        }
        if (!is_writable($path)) {
            echo "[WARN] Directorio no escribible: {$path}\n";
        } else {
            echo "[OK] Directorio escribible: {$path}\n";
        }
    }

    echo "Self test finalizado." . PHP_EOL;
    exit(0);
} catch (Throwable $e) {
    fwrite(STDERR, "[ERROR] " . $e->getMessage() . "\n");
    exit(1);
}

