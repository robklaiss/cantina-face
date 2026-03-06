<?php

require_once __DIR__ . '/../app/bootstrap.php';

$action = strtolower((string) input('action', 'stats'));
$currentUser = require_auth([ROLE_ADMIN, ROLE_STOCK, ROLE_CAJERA]);

switch ($action) {
    case 'stats':
        ensure_roles([ROLE_ADMIN, ROLE_STOCK, ROLE_CAJERA]);
        json_ok(backend_stats());
        break;

    case 'users':
        ensure_roles([ROLE_ADMIN]);
        handle_users();
        break;

    case 'user':
        ensure_roles([ROLE_ADMIN]);
        handle_user_detail();
        break;

    case 'user_reset':
        ensure_roles([ROLE_ADMIN]);
        handle_user_reset();
        break;

    case 'link_requests':
        ensure_roles([ROLE_ADMIN]);
        handle_link_requests();
        break;

    case 'link_request_decision':
        ensure_roles([ROLE_ADMIN]);
        handle_link_request_decision();
        break;

    case 'topups':
        ensure_roles([ROLE_ADMIN, ROLE_STOCK]);
        handle_topups();
        break;

    case 'topup_decision':
        ensure_roles([ROLE_ADMIN, ROLE_STOCK]);
        handle_topup_decision();
        break;

    case 'trigger_update':
        ensure_roles([ROLE_ADMIN]);
        handle_trigger_update();
        break;

    case 'update_status':
        ensure_roles([ROLE_ADMIN]);
        handle_update_status();
        break;

    default:
        json_error('Acción no soportada', 404);
}

function ensure_roles(array $roles): void
{
    $user = current_user();
    if (!$user || !in_array($user['role'], $roles, true)) {
        json_error('Permisos insuficientes', 403);
    }
}

function require_method(string $method): void
{
    if (http_method() !== strtoupper($method)) {
        json_error('Método no permitido', 405);
    }
}

function handle_users(): void
{
    if (http_method() === 'GET') {
        json_ok(list_users_admin());
        return;
    }

    if (http_method() === 'POST') {
        $payload = request_json();
        json_ok(create_user_admin($payload), 201);
        return;
    }

    json_error('Método no soportado', 405);
}

function handle_user_detail(): void
{
    $userId = parse_int(input('id'));
    if (!$userId) {
        json_error('id requerido', 422);
    }

    if (http_method() === 'GET') {
        $user = find_user_admin($userId);
        if (!$user) {
            json_error('Usuario no encontrado', 404);
        }
        json_ok($user);
        return;
    }

    if (in_array(http_method(), ['POST', 'PUT', 'PATCH'], true)) {
        $payload = request_json();
        json_ok(update_user_admin($userId, $payload));
        return;
    }

    json_error('Método no soportado', 405);
}

function handle_user_reset(): void
{
    require_method('POST');
    $userId = parse_int(input('id'));
    if (!$userId) {
        json_error('id requerido', 422);
    }
    $body = request_json();
    require_fields($body, ['new_password']);
    json_ok(reset_user_password_admin($userId, $body['new_password']));
}

function handle_link_requests(): void
{
    $status = sanitize_string(input('status'));
    $requests = list_link_requests_filtered(null, $status ?: null);
    json_ok($requests);
}

function handle_link_request_decision(): void
{
    require_method('POST');
    $requestId = parse_int(input('id'));
    if (!$requestId) {
        json_error('id requerido', 422);
    }
    $body = request_json();
    require_fields($body, ['decision']);
    $decision = validate_enum($body['decision'], ['approved', 'rejected'], 'decision');
    $studentId = isset($body['student_id']) ? (int) $body['student_id'] : null;
    $notes = sanitize_string($body['admin_notes'] ?? null);

    // Fetch request before updating so we have parent_id and student_name
    $request = db_fetch('SELECT * FROM link_requests WHERE id = :id', ['id' => $requestId]);

    update_link_request_status($requestId, $decision, $studentId, $notes);
    $updated = db_fetch('SELECT * FROM link_requests WHERE id = :id', ['id' => $requestId]);

    // Send approval email to parent
    if ($decision === 'approved' && $request) {
        $parent = find_user_by_id((int) $request['parent_id']);
        if ($parent && $parent['email']) {
            $firstName = $parent['first_name'] ?? ($parent['full_name'] ?? 'Padre/Madre');
            send_link_approved_email($parent['email'], $firstName, $request['student_name']);
        }
    }

    json_ok($updated);
}

function handle_topups(): void
{
    $status = sanitize_string(input('status'));
    json_ok(list_all_topups($status ?: null));
}

function handle_topup_decision(): void
{
    require_method('POST');
    $topupId = parse_int(input('id'));
    if (!$topupId) {
        json_error('id requerido', 422);
    }
    $body = request_json();
    require_fields($body, ['decision']);
    $decision = validate_enum($body['decision'], ['approved', 'rejected'], 'decision');
    $reference = sanitize_string($body['payment_reference'] ?? null);
    $notes = sanitize_string($body['admin_notes'] ?? null);
    $result = process_topup_decision($topupId, $decision, $reference, $notes);
    json_ok($result);
}

/**
 * Check all students linked to a parent and send low-credits email
 * if any balance is below the threshold. Called after any balance change.
 */
function maybe_send_low_credits_emails(int $parentId, float $threshold = 5000): void
{
    $parent = find_user_by_id($parentId);
    if (!$parent || !$parent['email']) {
        return;
    }
    $students = get_parent_students($parentId);
    foreach ($students as $student) {
        $balance = (float) ($student['balance'] ?? 0);
        if ($balance < $threshold) {
            $firstName = $parent['first_name'] ?? ($parent['full_name'] ?? 'Padre/Madre');
            send_low_credits_email($parent['email'], $firstName, $student['name'], $balance);
        }
    }
}

function handle_trigger_update(): void
{
    require_method('POST');
    $result = trigger_caja_update();
    if ($result['success']) {
        json_ok($result);
    } else {
        json_error($result['message'], 500);
    }
}

function handle_update_status(): void
{
    require_method('GET');
    $result = check_caja_update_status();
    if ($result['success']) {
        json_ok($result['data']);
    } else {
        json_error($result['message'], 500);
    }
}

