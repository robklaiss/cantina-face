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

$name  = trim((string) ($_GET['name'] ?? 'Alumno Test'));
$grade = trim((string) ($_GET['grade'] ?? '1A'));

$pdo = db();

$pdo->exec("
CREATE TABLE IF NOT EXISTS students (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  grade TEXT,
  photo_path TEXT,
  balance REAL DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now'))
);
");

$stmt = $pdo->prepare("INSERT INTO students (name, grade, balance) VALUES (:name, :grade, 0)");
$stmt->execute(['name' => $name, 'grade' => $grade]);

$id = (int) $pdo->lastInsertId();

json_ok([
    'created' => true,
    'student' => [
        'id' => $id,
        'name' => $name,
        'grade' => $grade,
    ],
]);
