<?php
// /api/_bootstrap_admin.php
// Crear un usuario ADMIN de emergencia. BORRAR este archivo después de usarlo.

require_once __DIR__ . '/../app/bootstrap.php';

header('Content-Type: application/json; charset=utf-8');

$email = $_GET['email'] ?? 'admin@siloe.com.py';
$pass  = $_GET['pass']  ?? 'Admin1234!';

try {
    $pdo = db();

    // Verifica si ya existe
    $st = $pdo->prepare("SELECT id, email, role FROM users WHERE email = :email LIMIT 1");
    $st->execute(['email' => $email]);
    $row = $st->fetch(PDO::FETCH_ASSOC);

    $hash = password_hash($pass, PASSWORD_DEFAULT);

    if ($row) {
        // Si existe, lo fuerza a ADMIN y resetea password
        $up = $pdo->prepare("UPDATE users SET role='admin', password_hash=:ph WHERE email=:email");
        $up->execute(['ph' => $hash, 'email' => $email]);

        echo json_encode([
            'ok' => true,
            'action' => 'updated_existing_user_to_admin',
            'email' => $email
        ], JSON_UNESCAPED_UNICODE|JSON_PRETTY_PRINT);
        exit;
    }

    // Crea nuevo ADMIN
    $ins = $pdo->prepare("
        INSERT INTO users (email, password_hash, role, created_at)
        VALUES (:email, :ph, 'admin', datetime('now'))
    ");
    $ins->execute(['email' => $email, 'ph' => $hash]);

    echo json_encode([
        'ok' => true,
        'action' => 'created_admin_user',
        'email' => $email
    ], JSON_UNESCAPED_UNICODE|JSON_PRETTY_PRINT);
} catch (Throwable $e) {
    error_log("[bootstrap_admin] " . $e->getMessage());
    http_response_code(500);
    echo json_encode([
        'ok' => false,
        'error' => $e->getMessage()
    ], JSON_UNESCAPED_UNICODE|JSON_PRETTY_PRINT);
}