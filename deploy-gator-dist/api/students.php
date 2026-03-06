<?php

require_once __DIR__ . '/../app/bootstrap.php';

$action = strtolower((string) input('action', 'list'));
$currentUser = require_auth([ROLE_ADMIN, ROLE_CAJERA, ROLE_STOCK]);

switch ($action) {
    case 'list':
        handle_student_list();
        break;
    case 'detail':
        handle_student_detail();
        break;
    case 'orders':
        handle_student_orders();
        break;
    case 'transactions':
        handle_student_transactions();
        break;
    case 'add_credits':
        ensure_roles([ROLE_ADMIN, ROLE_CAJERA]);
        handle_student_add_credits();
        break;
    case 'update':
        ensure_roles([ROLE_ADMIN]);
        handle_student_update();
        break;
    case 'delete':
        ensure_roles([ROLE_ADMIN]);
        handle_student_delete();
        break;
    case 'suggestions':
        handle_student_suggestions();
        break;
    case 'credit_history':
        handle_credit_history();
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

function require_student_id(): int
{
    $id = parse_int(input('id'));
    if (!$id) {
        json_error('id requerido', 422);
    }
    return $id;
}

function handle_student_list(): void
{
    $query = sanitize_string(input('query', ''));
    $limit = parse_int(input('limit', 100)) ?? 100;
    $rows = list_students_admin($query ?? '', $limit);
    $data = array_map('format_student_record', $rows);
    json_ok($data);
}

function handle_student_detail(): void
{
    $id = require_student_id();
    $student = find_student($id);
    if (!$student) {
        json_error('Alumno no encontrado', 404);
    }
    json_ok(format_student_record($student));
}

function handle_student_orders(): void
{
    $id = require_student_id();
    $status = sanitize_string(input('status_filter'));
    $orders = list_student_orders($id, $status ?: null);
    attach_products_to_order_items($orders);
    json_ok($orders);
}

function handle_student_transactions(): void
{
    $id = require_student_id();
    $limit = parse_int(input('limit', 50)) ?? 50;
    $rows = list_transactions_by_student($id, $limit);
    json_ok($rows);
}

function handle_student_add_credits(): void
{
    require_method(['POST', 'PUT']);
    $id = require_student_id();
    $body = request_json();
    require_fields($body, ['amount']);
    $amount = (float) $body['amount'];
    if ($amount <= 0) {
        json_error('Monto inválido', 422);
    }
    $student = adjust_student_balance($id, $amount, 'topup', ['source' => 'manual']);
    json_ok(['new_balance' => (float) $student['balance']]);
}

function handle_student_update(): void
{
    require_method(['POST', 'PUT']);
    $id = require_student_id();
    $payload = [
        'name' => sanitize_string($_POST['name'] ?? null),
        'grade' => sanitize_string($_POST['grade'] ?? null),
    ];
    if (isset($_POST['balance']) && $_POST['balance'] !== '') {
        $payload['balance'] = (float) $_POST['balance'];
    }

    if (!empty($_FILES['photo']) && is_uploaded_file($_FILES['photo']['tmp_name'])) {
        $payload['photo_path'] = save_student_photo($id, $_FILES['photo']);
    }

    $payload = array_filter($payload, function ($value) {
        return $value !== null && $value !== '';
    });

    if (!$payload) {
        json_error('No hay cambios para guardar', 422);
    }

    update_student($id, $payload);
    $student = find_student($id);
    json_ok(format_student_record($student));
}

function save_student_photo(int $studentId, array $file): string
{
    $dir = base_path('storage/uploads/students');
    if (!is_dir($dir)) {
        mkdir($dir, 0775, true);
    }
    $extension = pathinfo($file['name'], PATHINFO_EXTENSION) ?: 'jpg';
    $filename = 'student-' . $studentId . '-' . time() . '.' . strtolower($extension);
    $target = $dir . '/' . $filename;
    if (!move_uploaded_file($file['tmp_name'], $target)) {
        json_error('No se pudo guardar la foto', 500);
    }
    return 'uploads/students/' . $filename;
}

function handle_student_delete(): void
{
    require_method(['DELETE', 'POST']);
    $id = require_student_id();
    delete_student($id);
    json_ok(['deleted' => true]);
}

function handle_student_suggestions(): void
{
    json_ok([]);
}

function handle_credit_history(): void
{
    $id = require_student_id();
    $limit = parse_int(input('limit', 25)) ?? 25;
    $rows = list_transactions_by_student($id, $limit);
    json_ok($rows);
}

function require_method($allowed): void
{
    $allowed = (array) $allowed;
    if (!in_array(http_method(), array_map('strtoupper', $allowed), true)) {
        json_error('Método no permitido', 405, ['allowed' => $allowed]);
    }
}

