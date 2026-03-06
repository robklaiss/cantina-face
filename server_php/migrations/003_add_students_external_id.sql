ALTER TABLE students
    ADD COLUMN external_id VARCHAR(191) NULL AFTER id;

ALTER TABLE students
    ADD UNIQUE INDEX uq_students_external_id (external_id);
