<?php
return [
  'app' => [
    'name' => 'Cantina Face',
    'env'  => 'production',
    'timezone' => 'America/Asuncion',
  ],

  'session' => [
    'name' => 'CANTINASESSID',
    'lifetime' => 3600,
  ],

  'caja' => [
    'url' => 'http://localhost:8000',
    'internal_token' => 'cantina-update-secret-2026',
  ],

  'db' => [
    'driver' => 'sqlite',
    'path' => __DIR__ . '/../data/db.sqlite',
  ],
];