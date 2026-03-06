PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL UNIQUE,
  full_name TEXT,
  first_name TEXT,
  last_name TEXT,
  dni TEXT,
  phone TEXT,
  role TEXT NOT NULL DEFAULT 'parent',
  point_of_sale_id INTEGER,
  password_hash TEXT NOT NULL,
  plain_password TEXT,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT
);

-- STUDENTS
CREATE TABLE IF NOT EXISTS students (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  grade TEXT,
  balance REAL NOT NULL DEFAULT 0,
  photo_path TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS parent_student (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id INTEGER NOT NULL,
  student_id INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(parent_id, student_id),
  FOREIGN KEY(parent_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
);

-- TRANSACTIONS
CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id INTEGER NOT NULL,
  amount REAL NOT NULL,
  txn_type TEXT NOT NULL,
  meta_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
);

-- PRODUCTS
CREATE TABLE IF NOT EXISTS products (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  price REAL NOT NULL DEFAULT 0,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- TOPUP REQUESTS
CREATE TABLE IF NOT EXISTS topup_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id INTEGER NOT NULL,
  total_amount REAL NOT NULL,
  allocation_mode TEXT NOT NULL,
  allocations_json TEXT,
  payment_reference TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(parent_id) REFERENCES users(id) ON DELETE CASCADE
);

-- AUDIT LOG
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  action TEXT NOT NULL,
  entity TEXT,
  payload TEXT,
  ip_address TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- LINK REQUESTS (padres solicitan vinculación)
CREATE TABLE IF NOT EXISTS link_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id INTEGER NOT NULL,
  student_identifier TEXT,
  student_name TEXT NOT NULL,
  student_grade TEXT,
  notes TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  admin_notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(parent_id) REFERENCES users(id) ON DELETE CASCADE
);

-- SCHEDULED ORDERS
CREATE TABLE IF NOT EXISTS scheduled_orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id INTEGER NOT NULL,
  student_id INTEGER NOT NULL,
  scheduled_for TEXT NOT NULL,
  notes TEXT,
  pay_from_balance INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(parent_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scheduled_order_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id INTEGER NOT NULL,
  product_id INTEGER NOT NULL,
  quantity INTEGER NOT NULL DEFAULT 1,
  FOREIGN KEY(order_id) REFERENCES scheduled_orders(id) ON DELETE CASCADE,
  FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE RESTRICT
);

-- DAILY MENU
CREATE TABLE IF NOT EXISTS daily_menu (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  menu_date TEXT NOT NULL,
  title TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(menu_date)
);

CREATE TABLE IF NOT EXISTS daily_menu_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  menu_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  meal_type TEXT,
  FOREIGN KEY(menu_id) REFERENCES daily_menu(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS menu_selections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  menu_id INTEGER NOT NULL,
  menu_item_id INTEGER NOT NULL,
  student_id INTEGER NOT NULL,
  parent_id INTEGER NOT NULL,
  menu_date TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(menu_id) REFERENCES daily_menu(id) ON DELETE CASCADE,
  FOREIGN KEY(menu_item_id) REFERENCES daily_menu_items(id) ON DELETE CASCADE,
  FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
  FOREIGN KEY(parent_id) REFERENCES users(id) ON DELETE CASCADE
);

-- PASSWORD RESET TOKENS
CREATE TABLE IF NOT EXISTS password_reset_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  token TEXT NOT NULL UNIQUE,
  expires_at TEXT NOT NULL,
  used INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- SEED opcional: si querés un admin por default (cambiá email/hash)
-- INSERT INTO users(email,password_hash,role,full_name) VALUES('admin@local', '$2y$10$....', 'admin', 'Admin');
