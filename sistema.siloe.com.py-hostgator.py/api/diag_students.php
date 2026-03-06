<?php
header('Content-Type: application/json; charset=utf-8');

function fail($msg, $extra = []) {
  http_response_code(500);
  echo json_encode(array_merge(['ok'=>false,'error'=>$msg], $extra), JSON_UNESCAPED_UNICODE|JSON_PRETTY_PRINT);
  exit;
}

// Ajustá esta ruta si tu config está en otro lado:
$config_path = __DIR__ . '/../app/config.php';
if (!file_exists($config_path)) fail('config.php no encontrado', ['config_path'=>$config_path]);

$cfg = require $config_path;
if (!is_array($cfg)) fail('config.php no devolvió un array');

$db = $cfg['db'] ?? null;
if (!$db || ($db['driver'] ?? '') !== 'sqlite') fail('db config no es sqlite', ['db'=>$db]);

$db_path = $db['path'] ?? '';
if (!$db_path) fail('db.path vacío');
if (!file_exists($db_path)) fail('sqlite no existe', ['db_path'=>$db_path]);

try {
  $pdo = new PDO('sqlite:' . $db_path);
  $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

  // Intentar detectar tabla (students vs alumnos)
  $tables = $pdo->query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")->fetchAll(PDO::FETCH_COLUMN);

  $counts = [];
  foreach ($tables as $t) {
    if (preg_match('/student|alumn/i', $t)) {
      $counts[$t] = (int)$pdo->query("SELECT COUNT(*) FROM \"$t\"")->fetchColumn();
    }
  }

  echo json_encode([
    'ok' => true,
    'config_path' => $config_path,
    'db_path' => $db_path,
    'db_size_bytes' => filesize($db_path),
    'tables' => $tables,
    'counts_like_students' => $counts,
  ], JSON_UNESCAPED_UNICODE|JSON_PRETTY_PRINT);

} catch (Throwable $e) {
  fail('exception', ['message'=>$e->getMessage()]);
}