<?php

require_once __DIR__ . '/../app/bootstrap.php';

$action = strtolower((string) input('action', http_method() === 'GET' ? 'me' : 'login'));

switch ($action) {
    case 'login':
        handle_login();
        break;
    case 'logout':
        handle_logout();
        break;
    case 'me':
        handle_me();
        break;
    case 'register':
        handle_register();
        break;
    case 'forgot_password':
        handle_forgot_password();
        break;
    case 'reset_password':
        handle_reset_password();
        break;
    default:
        json_error('Acción no soportada', 404);
}

function require_method(string $method): void
{
    if (http_method() !== strtoupper($method)) {
        json_error('Método no permitido', 405);
    }
}

function handle_login(): void
{
    require_method('POST');
    rate_limit_check('login:' . get_client_ip(), 30, 60);

    $body = request_body();
    $email = $body['username'] ?? $body['email'] ?? input('username', input('email', ''));
    $password = $body['password'] ?? input('password');

    if (!$email || !$password) {
        json_error('Credenciales requeridas', 422);
    }

    $user = authenticate_credentials($email, $password);
    $session = issue_session_for_user($user);
    json_ok($session);
}

function handle_logout(): void
{
    require_method('POST');
    if (session_user()) {
        require_csrf_token();
        session_logout();
    }
    json_ok(['message' => 'Sesión finalizada']);
}

function handle_me(): void
{
    require_method('GET');
    $user = require_auth();
    json_ok([
        'user' => $user,
        'csrf_token' => $_SESSION['csrf_token'] ?? null,
    ]);
}

function handle_register(): void
{
    require_method('POST');
    rate_limit_check('register:' . get_client_ip(), 10, 300);

    $body = request_json();
    $email    = trim((string) ($body['email'] ?? ''));
    $password = (string) ($body['password'] ?? '');
    $firstName = sanitize_string($body['first_name'] ?? null) ?? '';
    $lastName  = sanitize_string($body['last_name'] ?? null) ?? '';
    $phone     = sanitize_string($body['phone'] ?? null);
    $dni       = sanitize_string($body['dni'] ?? null);

    if (!$email || !$password) {
        json_error('Email y contraseña son requeridos', 422);
    }
    $email = validate_email($email);
    validate_password($password);

    if (find_user_by_email($email)) {
        json_error('El correo ya está registrado', 422);
    }

    $fullName = trim("$firstName $lastName") ?: $email;

    $stmt = db()->prepare(
        'INSERT INTO users (email, password_hash, role, full_name, first_name, last_name, phone, dni, is_active)
         VALUES (:email, :hash, :role, :full_name, :first_name, :last_name, :phone, :dni, 1)'
    );
    $stmt->execute([
        'email'      => $email,
        'hash'       => hash_password($password),
        'role'       => ROLE_PARENT,
        'full_name'  => $fullName,
        'first_name' => $firstName ?: null,
        'last_name'  => $lastName ?: null,
        'phone'      => $phone,
        'dni'        => $dni,
    ]);
    $userId = (int) db()->lastInsertId();

    // Optionally store pending link requests submitted during registration
    $pendingKids = $body['kids'] ?? [];
    if (is_array($pendingKids) && $pendingKids) {
        foreach ($pendingKids as $kid) {
            $kidName = sanitize_string($kid['name'] ?? null);
            if (!$kidName) {
                continue;
            }
            create_link_request([
                'parent_id'          => $userId,
                'student_identifier' => sanitize_string($kid['identifier'] ?? null),
                'student_name'       => $kidName,
                'student_grade'      => sanitize_string($kid['grade'] ?? null),
                'notes'              => sanitize_string($kid['notes'] ?? null),
            ]);
        }
    }

    $displayName = $firstName ?: ($fullName ?: $email);
    send_parent_registration_received_email($email, $displayName);

    json_ok(['message' => 'Registro exitoso. Revisaremos la vinculación con tus hijos y te avisaremos cuando puedas gestionar saldos.'], 201);
}

function handle_forgot_password(): void
{
    require_method('POST');
    rate_limit_check('forgot:' . get_client_ip(), 5, 300);

    $body  = request_json();
    $email = trim((string) ($body['email'] ?? ''));
    if (!$email) {
        json_error('Email requerido', 422);
    }

    // Always respond OK to avoid user enumeration
    $user = find_user_by_email($email);
    if ($user) {
        $token     = bin2hex(random_bytes(32));
        $expiresAt = date('Y-m-d H:i:s', time() + 3600);

        // Invalidate previous tokens for this user
        db()->prepare('UPDATE password_reset_tokens SET used = 1 WHERE user_id = :uid')
             ->execute(['uid' => $user['id']]);

        db()->prepare(
            'INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (:uid, :token, :exp)'
        )->execute(['uid' => $user['id'], 'token' => $token, 'exp' => $expiresAt]);

        $appUrl   = config('app.url', '');
        $resetUrl = rtrim($appUrl, '/') . '/padres/?action=reset_password&token=' . $token;
        $firstName = $user['first_name'] ?? ($user['full_name'] ?? 'Padre/Madre');
        send_password_reset_email($user['email'], $firstName, $resetUrl);
    }

    json_ok(['message' => 'Si el correo existe, recibirás un enlace para restablecer tu contraseña.']);
}

function handle_reset_password(): void
{
    require_method('POST');
    rate_limit_check('reset:' . get_client_ip(), 10, 300);

    $body     = request_body();
    $token    = trim((string) ($body['token'] ?? ''));
    $password = (string) ($body['password'] ?? '');

    if (!$token || !$password) {
        json_error('Token y nueva contraseña son requeridos', 422);
    }
    validate_password($password);

    $pdo  = db();
    $row  = $pdo->prepare(
        'SELECT * FROM password_reset_tokens WHERE token = :token AND used = 0 LIMIT 1'
    );
    $row->execute(['token' => $token]);
    $record = $row->fetch();

    if (!$record) {
        json_error('Token inválido o ya utilizado', 400);
    }
    if (strtotime($record['expires_at']) < time()) {
        json_error('El enlace ha expirado. Solicitá uno nuevo.', 400);
    }

    $hash = hash_password($password);
    $pdo->prepare('UPDATE users SET password_hash = :hash, updated_at = datetime("now") WHERE id = :id')
        ->execute(['hash' => $hash, 'id' => $record['user_id']]);

    $pdo->prepare('UPDATE password_reset_tokens SET used = 1 WHERE id = :id')
        ->execute(['id' => $record['id']]);

    json_ok(['message' => 'Contraseña actualizada correctamente. Ya podés iniciar sesión.']);
}

