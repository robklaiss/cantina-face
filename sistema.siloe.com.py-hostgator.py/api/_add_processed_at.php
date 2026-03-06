<?php
require_once __DIR__ . '/../app/bootstrap.php';

header('Content-Type: application/json; charset=utf-8');

try {
    $pdo = db();

    $cols = $pdo->query("PRAGMA table_info(link_requests)")->fetchAll(PDO::FETCH_ASSOC);
    $hasProcessedAt = false;
    foreach ($cols as $col) {
        if (($col['name'] ?? '') === 'processed_at') {
            $hasProcessedAt = true;
            break;
        }
    }

    if (!$hasProcessedAt) {
        $pdo->exec("ALTER TABLE link_requests ADD COLUMN processed_at TEXT NULL");
    }

    echo json_encode([
        'ok' => true,
        'added' => !$hasProcessedAt,
        'column' => 'processed_at',
    ], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);

} catch (Throwable $e) {
    http_response_code(500);
    echo json_encode([
        'ok' => false,
        'error' => $e->getMessage(),
        'file' => $e->getFile(),
        'line' => $e->getLine(),
    ], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
}