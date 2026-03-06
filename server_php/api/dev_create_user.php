<?php

declare(strict_types=1);

require_once __DIR__ . '/../app/bootstrap_core.php';
require_once __DIR__ . '/../app/auth/login.php';

$cfg = app_get('config', []);
$keyExpected = (string) (($cfg['dev']['init_key'] ?? ''));

$key = (string) ($_GET['key'] ?? '');
if ($keyExpected === '' || !hash_equals($keyExpected, $key)) {
    http_response_code(404);
    echo 'Not Found';
    exit;
}

$email = strtolower(trim((string) ($_GET['email'] ?? 'test@test.com')));
$password = (string) ($_GET['password'] ?? '1234');
$role = (string) ($_GET['role'] ?? ROLE_PARENT);

try {
    $pdo = db();

    $cols = $pdo->query('PRAGMA table_info(users)')->fetchAll(PDO::FETCH_ASSOC);
    $colNames = array_map(fn ($c) => $c['name'], $cols);

    $passCol = null;
    if (in_array('password_hash', $colNames, true)) {
        $passCol = 'password_hash';
    }
    if ($passCol === null && in_array('password', $colNames, true)) {
        $passCol = 'password';
    }

    if ($passCol === null) {
        json_error('No password column found in users table (expected password_hash or password).', 500, [
            'columns' => $colNames,
        ]);
    }

    $stmt = $pdo->prepare('SELECT id,email,role FROM users WHERE lower(email)=:email LIMIT 1');
    $stmt->execute(['email' => $email]);
    $existing = $stmt->fetch(PDO::FETCH_ASSOC);
    if ($existing) {
        json_ok(['created' => false, 'user' => $existing]);
    }

    $hash = hash_password($password);

    $hasFirst = in_array('first_name', $colNames, true);
    $hasFull = in_array('full_name', $colNames, true);

    $sqlCols = "email, {$passCol}, role";
    $sqlVals = ':email, :pass, :role';
    $params = [
        ':email' => $email,
        ':pass' => $hash,
        ':role' => $role,
    ];

    if ($hasFirst) {
        $sqlCols .= ', first_name';
        $sqlVals .= ', :first_name';
        $params[':first_name'] = 'Test';
    } elseif ($hasFull) {
        $sqlCols .= ', full_name';
        $sqlVals .= ', :full_name';
        $params[':full_name'] = 'Test Parent';
    }

    $pdo->prepare("INSERT INTO users ({$sqlCols}) VALUES ({$sqlVals})")->execute($params);

    $id = (int) $pdo->lastInsertId();
    $stmt = $pdo->prepare('SELECT id,email,role FROM users WHERE id=:id');
    $stmt->execute(['id' => $id]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    json_ok([
        'created' => true,
        'user' => $user,
        'login' => ['email' => $email, 'password' => $password],
    ]);
} catch (Throwable $e) {
    json_error('Dev create failed', 500, ['detail' => $e->getMessage()]);
}
