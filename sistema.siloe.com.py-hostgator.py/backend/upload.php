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

if (!isset($_FILES['zipfile'])) {
    respond_json([
        'ok' => false,
        'error' => 'zipfile field is required',
    ], 400);
}

$upload = $_FILES['zipfile'];

if (!is_array($upload) || ($upload['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_OK) {
    $errorCode = $upload['error'] ?? UPLOAD_ERR_NO_FILE;
    $messages = [
        UPLOAD_ERR_INI_SIZE => 'File exceeds server limit',
        UPLOAD_ERR_FORM_SIZE => 'File exceeds form limit',
        UPLOAD_ERR_PARTIAL => 'Partial upload detected',
        UPLOAD_ERR_NO_FILE => 'No file uploaded',
        UPLOAD_ERR_NO_TMP_DIR => 'Missing temp directory',
        UPLOAD_ERR_CANT_WRITE => 'Failed to write file to disk',
        UPLOAD_ERR_EXTENSION => 'Upload stopped by extension',
    ];
    respond_json([
        'ok' => false,
        'error' => $messages[$errorCode] ?? 'Upload failed',
    ], 400);
}

$originalName = (string) ($upload['name'] ?? '');
$ext = strtolower(pathinfo($originalName, PATHINFO_EXTENSION));

if ($ext !== 'zip') {
    respond_json([
        'ok' => false,
        'error' => 'Only .zip files are allowed',
    ], 400);
}

$timestamp = gmdate('Ymd_His');
$finalName = "project_{$timestamp}.zip";
$destination = RELEASES_DIR . '/' . $finalName;

if (!is_uploaded_file($upload['tmp_name'])) {
    respond_json([
        'ok' => false,
        'error' => 'Invalid upload source',
    ], 400);
}

if (!move_uploaded_file($upload['tmp_name'], $destination)) {
    respond_json([
        'ok' => false,
        'error' => 'Failed to move uploaded file',
    ], 500);
}

chmod($destination, 0644);

$meta = release_metadata($finalName);

if ($meta === null) {
    respond_json([
        'ok' => false,
        'error' => 'Unable to inspect uploaded file',
    ], 500);
}

respond_json([
    'ok' => true,
    'release' => $meta,
]);
