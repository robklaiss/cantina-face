<?php

declare(strict_types=1);

function require_fields(array $data, array $fields): void
{
    foreach ($fields as $field) {
        if (!isset($data[$field]) || $data[$field] === '' || $data[$field] === null) {
            json_error("Campo requerido: {$field}", 422);
        }
    }
}

function validate_email(string $email): string
{
    $email = strtolower(trim($email));
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        json_error('Email inválido', 422);
    }
    return $email;
}

function validate_password(string $password): string
{
    if (strlen($password) < 6) {
        json_error('La contraseña debe tener al menos 6 caracteres', 422);
    }
    return $password;
}

function validate_enum(string $value, array $allowed, string $field = 'valor'): string
{
    if (!in_array($value, $allowed, true)) {
        json_error("{$field} inválido", 422, ['allowed' => $allowed]);
    }
    return $value;
}

function boolval_str($value): bool
{
    if (is_bool($value)) {
        return $value;
    }
    return in_array(strtolower((string) $value), ['1', 'true', 'yes', 'si', 'on'], true);
}

