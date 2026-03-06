<?php

/**
 * Cantina Face Configuration
 * Copy this file to app/config.php and edit for your environment
 */
 
'caja' => [
    'url' => 'http://localhost:8000',  // o la URL pública de la caja si la tenés
    'internal_token' => 'cantina-update-secret-2026',  // debe coincidir con el .env de la caja
],

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

    // SQLite configuration (default - recommended for HostGator)
    'db' => [
        'driver' => 'sqlite',
        'path' => __DIR__ . '/../data/db.sqlite',
    ],

    // MySQL configuration (alternative)
    // 'db' => [
    //     'driver' => 'mysql',
    //     'host' => 'localhost',
    //     'port' => 3306,
    //     'database' => 'cantina_face',
    //     'username' => 'cantina_user',
    //     'password' => 'super-secret',
    //     'charset' => 'utf8mb4',
    //     'collation' => 'utf8mb4_unicode_ci',
    // ],

    // Mail / SMTP configuration
    // Set driver to 'smtp' and fill credentials to enable real email sending.
    // Until then, emails are written to the app log (driver = 'log').
    'mail' => [
        'driver'       => 'log',          // 'log' | 'smtp'
        'host'         => '',             // e.g. mail.hostgator.com
        'port'         => 587,            // 587 (STARTTLS) or 465 (SSL)
        'encryption'   => 'tls',          // 'tls' | 'ssl' | ''
        'username'     => '',             // SMTP username / email address
        'password'     => '',             // SMTP password
        'from_address' => '',             // e.g. noreply@siloe.com.py
        'from_name'    => 'Cantina Siloe',
    ],

    'dev' => [
        'init_key' => 'INIT123',
    ],
];
