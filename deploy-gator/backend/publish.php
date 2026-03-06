<?php
require __DIR__ . '/config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    respond_json([
        'ok' => false,
        'error' => 'Method not allowed',
    ], 405);
}

$token = $_POST['token'] ?? null;
verify_token($token);

$requested = trim((string) ($_POST['filename'] ?? ''));
$filename = basename($requested);

if ($filename === '' || !preg_match('/^project_\d{8}_\d{6}\.zip$/', $filename)) {
    respond_json([
        'ok' => false,
        'error' => 'Invalid filename',
    ], 400);
}

$meta = release_metadata($filename);

if ($meta === null) {
    respond_json([
        'ok' => false,
        'error' => 'Release not found',
    ], 404);
}

write_latest($meta);

respond_json([
    'ok' => true,
    'release' => $meta,
]);
