<?php

declare(strict_types=1);

/**
 * Cantina Face - Bootstrap (Full)
 * HostGator-ready: shared hosting, Apache + PHP 8.3
 *
 * This bootstrap loads the core + all components for full API functionality.
 * Includes session, auth, validators, domain logic.
 */

// Load core bootstrap (config, db, response, logger, error handlers)
require_once __DIR__ . '/bootstrap_core.php';

$appDir = __DIR__;

// Load remaining app components
require_once $appDir . '/session.php';
require_once $appDir . '/validators.php';
require_once $appDir . '/auth/login.php';
require_once $appDir . '/auth/middleware.php';
require_once $appDir . '/domain.php';
require_once $appDir . '/mailer.php';

// Start session
session_bootstrap();
