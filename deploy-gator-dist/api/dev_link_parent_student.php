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

$parentId  = (int) ($_GET['parent_id'] ?? 0);
$studentId = (int) ($_GET['student_id'] ?? 0);

if ($parentId <= 0 || $studentId <= 0) {
    json_error('parent_id y student_id requeridos', 422);
}

$pdo = db();

// Create table if missing (safety)
$pdo->exec("
CREATE TABLE IF NOT EXISTS parent_student (
  parent_id INTEGER NOT NULL,
  student_id INTEGER NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (parent_id, student_id)
);
");

// Idempotente
$stmt = $pdo->prepare("INSERT OR IGNORE INTO parent_student (parent_id, student_id) VALUES (:p, :s)");
$stmt->execute(['p' => $parentId, 's' => $studentId]);

json_ok([
    'linked' => true,
    'parent_id' => $parentId,
    'student_id' => $studentId,
]);
