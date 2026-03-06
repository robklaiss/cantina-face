<?php

declare(strict_types=1);

require_once __DIR__ . '/../app/bootstrap_core.php';

$cfg = app_get('config', []);
$keyExpected = (string) (($cfg['dev']['init_key'] ?? ''));

$key = (string) ($_GET['key'] ?? '');
if ($keyExpected === '' || !hash_equals($keyExpected, $key)) {
    http_response_code(404);
    echo 'Not Found';
    exit;
}

try {
    $pdo = db();

    $schema = $pdo->query('PRAGMA table_info(users)')->fetchAll(PDO::FETCH_ASSOC);
    $count = (int) $pdo->query('SELECT COUNT(*) AS c FROM users')->fetchColumn();

    json_ok([
        'users_count' => $count,
        'users_schema' => $schema,
    ]);
} catch (Throwable $e) {
    json_error('Dev dump failed', 500, ['detail' => $e->getMessage()]);
}
