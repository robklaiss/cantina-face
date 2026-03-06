<?php

declare(strict_types=1);

require_once __DIR__ . '/helpers.php';
require_once __DIR__ . '/logger.php';

function db(): PDO
{
    $pdo = app_get('db');
    if ($pdo instanceof PDO) {
        return $pdo;
    }

    $config = config('db');
    if (!$config) {
        throw new RuntimeException('Database configuration missing');
    }

    $dsn = sprintf(
        '%s:host=%s;port=%d;dbname=%s;charset=%s',
        $config['driver'] ?? 'mysql',
        $config['host'] ?? 'localhost',
        $config['port'] ?? 3306,
        $config['database'] ?? '',
        $config['charset'] ?? 'utf8mb4'
    );

    $options = [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES => false,
    ];

    $pdo = new PDO($dsn, $config['username'] ?? '', $config['password'] ?? '', $options);
    app_set('db', $pdo);
    return $pdo;
}

function db_query(string $sql, array $params = []): PDOStatement
{
    $stmt = db()->prepare($sql);
    foreach ($params as $key => $value) {
        $paramKey = is_int($key) ? $key + 1 : (is_string($key) && $key[0] !== ':' ? ':' . $key : $key);
        $stmt->bindValue($paramKey, $value);
    }
    $stmt->execute();
    return $stmt;
}

function db_fetch(string $sql, array $params = []): ?array
{
    $stmt = db_query($sql, $params);
    $row = $stmt->fetch();
    return $row === false ? null : $row;
}

function db_fetch_all(string $sql, array $params = []): array
{
    return db_query($sql, $params)->fetchAll();
}

function db_insert(string $sql, array $params = []): int
{
    $stmt = db_query($sql, $params);
    if ($stmt->rowCount() <= 0) {
        return 0;
    }
    return (int) db()->lastInsertId();
}

function db_execute(string $sql, array $params = []): int
{
    $stmt = db_query($sql, $params);
    return $stmt->rowCount();
}

function db_transaction(callable $callback)
{
    $pdo = db();
    try {
        $pdo->beginTransaction();
        $result = $callback($pdo);
        $pdo->commit();
        return $result;
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) {
            $pdo->rollBack();
        }
        log_error('DB transaction failed', ['error' => $e->getMessage()]);
        throw $e;
    }
}

