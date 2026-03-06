# Performance Notes — Cantina Face Checkout

## Problem

After selecting a recognized student, the sales UI was slow to become interactive.
The checkout screen loads data from three endpoints in sequence:

1. `GET /api/students/{id}` — student detail (name, balance, photo)
2. `GET /api/students/{id}/scheduled-orders?status_filter=pending` — pending scheduled orders
3. `GET /api/products` — product catalog

## Root Causes Identified

### 1. `/api/products` hit the DB on every call
The product catalog rarely changes during a shift, yet every checkout loaded it
fresh from SQLite. On slow Chromebook storage this added 5–15 ms per call,
multiplied by every student selection.

**Fix:** Added an in-memory products cache with a configurable TTL
(`PRODUCTS_CACHE_TTL`, default 5 seconds). The cache is invalidated on
product create/update/seed. Subsequent calls within the TTL window return
instantly without touching the DB.

### 2. Scheduled orders query was already indexed but lacked timing visibility
The `scheduledorder(student_id, status)` composite index was already present.
However, there was no way to diagnose slowness without enabling verbose logging.

**Fix:** Added `PERF_LOG=1` environment variable. When set, the `cantina.perf`
logger emits detailed timing for every stage: products query, student detail,
scheduled orders (broken down into orders query, items query, serialization).
When `PERF_LOG` is off (default), these logs are suppressed to WARNING level.

### 3. Face recognition CPU saturation affected all endpoints
On low-power hardware, continuous face embedding inference saturated the CPU,
causing all HTTP responses (including checkout data) to slow down.

**Already mitigated (prior work):**
- `FACE_MAX_EMB_PER_SEC=2` — limits embeddings per second
- `FACE_CACHE_MS=500` — reuses recent embeddings when the face hasn't changed
- `ORT_INTRA_THREADS=1` / `ORT_INTER_THREADS=1` — prevents ONNX from spawning extra threads
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, etc. — limits math library threads

These are all exported by `deploy/run.sh` and tunable via environment variables.

## Indexes

All relevant indexes were already in place:

| Index | Table | Columns |
|-------|-------|---------|
| `idx_scheduledorder_student_status` | `scheduledorder` | `student_id, status` |
| `idx_transaction_student` | `transaction` | `student_id` |
| `idx_transaction_created_at` | `transaction` | `created_at` |
| `idx_productstock_product_pos` | `productstock` | `product_id, point_of_sale_id` |
| `idx_student_point_of_sale` | `student` | `point_of_sale_id` |

No N+1 query issues were found in the checkout path — scheduled order items
and products are fetched in bulk using `IN(...)` clauses.

## How to Diagnose

```bash
# Enable perf logging
PERF_LOG=1 ./deploy/run.sh

# Check timing summary (requires auth)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/health/timing

# Run the checkout benchmark
python scripts/bench_checkout.py --username admin@siloe.com.py --password admin321 --iterations 5
```

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `PERF_LOG` | `0` | Set to `1` to enable detailed timing logs |
| `PRODUCTS_CACHE_TTL` | `5` | Seconds to cache `/api/products` results |
| `FACE_MAX_EMB_PER_SEC` | `2` | Max face embeddings per second |
| `FACE_CACHE_MS` | `500` | Reuse window for identical face embeddings |
| `ORT_INTRA_THREADS` | `1` | ONNX intra-op threads |
| `ORT_INTER_THREADS` | `1` | ONNX inter-op threads |

## Before / After

| Metric | Before | After |
|--------|--------|-------|
| `/api/products` (cached) | 5–15 ms | <0.1 ms |
| `/api/products` (cold) | 5–15 ms | 5–15 ms (same, but rare) |
| CPU during idle checkout | ~60–100% (face loop) | ~10–30% (throttled) |
| Perf visibility | None | `PERF_LOG=1` logs all stages |
