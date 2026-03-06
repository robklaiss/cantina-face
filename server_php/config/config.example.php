<?php

return [
    'app' => [
        'name' => 'Cantina Face',
        'env' => 'production',
        'url' => 'https://tu-dominio.com',
        'log_path' => __DIR__ . '/../storage/logs/app-' . date('Y-m-d') . '.log',
        'timezone' => 'America/Asuncion',
        'internal_token' => 'set-unique-internal-token',
    ],

    'session' => [
        'name' => 'CANTINASESSID',
        'lifetime' => 3600,
    ],

    'db' => [
        'driver' => 'mysql',
        'host' => 'localhost',
        'port' => 3306,
        'database' => 'cantina_face',
        'username' => 'cantina_user',
        'password' => 'super-secret',
        'charset' => 'utf8mb4',
        'collation' => 'utf8mb4_unicode_ci',
    ],
];
