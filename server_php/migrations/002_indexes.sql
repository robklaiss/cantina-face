-- Cantina Face - Secondary indexes for common queries
-- Ejecutar después de 001_init.sql

ALTER TABLE users
    ADD INDEX idx_users_role (role),
    ADD INDEX idx_users_created_at (created_at);

ALTER TABLE students
    ADD INDEX idx_students_name (name),
    ADD INDEX idx_students_grade (grade),
    ADD INDEX idx_students_created_at (created_at);

ALTER TABLE parent_student
    ADD INDEX idx_parent_student_parent (parent_id),
    ADD INDEX idx_parent_student_student (student_id);

ALTER TABLE products
    ADD INDEX idx_products_name (name);

ALTER TABLE transactions
    ADD INDEX idx_transactions_student_created (student_id, created_at),
    ADD INDEX idx_transactions_created (created_at),
    ADD INDEX idx_transactions_type (txn_type);

ALTER TABLE topup_requests
    ADD INDEX idx_topups_parent (parent_id),
    ADD INDEX idx_topups_status_created (status, created_at);

ALTER TABLE scheduled_orders
    ADD INDEX idx_sched_orders_parent (parent_id),
    ADD INDEX idx_sched_orders_student (student_id),
    ADD INDEX idx_sched_orders_status_date (status, scheduled_for);

ALTER TABLE scheduled_order_items
    ADD INDEX idx_sched_order_items_order (order_id),
    ADD INDEX idx_sched_order_items_product (product_id);

ALTER TABLE menu_selections
    ADD INDEX idx_menu_selections_parent (parent_id),
    ADD INDEX idx_menu_selections_student (student_id),
    ADD INDEX idx_menu_selections_date (menu_date);

ALTER TABLE link_requests
    ADD INDEX idx_link_requests_parent (parent_id),
    ADD INDEX idx_link_requests_status (status);

ALTER TABLE audit_log
    ADD INDEX idx_audit_log_user (user_id),
    ADD INDEX idx_audit_log_action (action),
    ADD INDEX idx_audit_log_created_at (created_at);

