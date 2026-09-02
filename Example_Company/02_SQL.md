# SQL

Not mentioned in the referral directly, but **heavily and consistently emphasized** across independent sources for this role — enough that skipping it would be a mistake. 🟢

---

### Question 1 — Window Functions

> Example Company operates at petabyte scale. Interviewers reportedly lean hard on window functions: `ROW_NUMBER`, `RANK`, `PERCENTILE_CONT`, `LAG`/`LEAD`, and non-trivial frame specs. (Source: search-summarized 1point3acres content, corroborated independently by a second summary) 🟡

**Practice prompt**: "For each product, find the day-over-day change in price, and flag any product whose price dropped more than 20% in a single day."

```sql
WITH price_changes AS (
  SELECT
    product_id,
    price_date,
    price,
    LAG(price) OVER (PARTITION BY product_id ORDER BY price_date) AS prev_price
  FROM product_prices
)
SELECT
  product_id,
  price_date,
  price,
  prev_price,
  (price - prev_price) / prev_price AS pct_change
FROM price_changes
WHERE prev_price IS NOT NULL
  AND (price - prev_price) / prev_price <= -0.20;
```

Know cold: the difference between `ROW_NUMBER` (unique rank, ties broken arbitrarily), `RANK` (ties share rank, gaps after), and `DENSE_RANK` (ties share rank, no gaps) — this is a classic follow-up.

---

### Question 2 — De-duplication Under Retries (dedup reasoning, catalog/event-quality flavored)

> An experiment readout shows a drop in add-to-cart rate for search traffic. You suspect duplicate `add_to_cart` events caused by client-side retries. Given `events(user_id, event_ts, event_name, request_id, sku_id, session_id)`, compute the daily add-to-cart rate per search session, correctly de-duplicated.

**Why this is a good practice proxy for this team**: it's not confirmed verbatim from a leaked interview, but it's exactly the kind of data-quality reasoning a **catalog/trust-and-safety** team lives in — duplicate signals, noisy client events, and picking the right denominator. Treat it as strong practice even though the exact source URL couldn't be confirmed. 🔴

**Answer approach**

Two traps to name explicitly before writing SQL:
1. `COUNT(*)` on raw events double-counts retried `add_to_cart` calls that share a `request_id`.
2. `COUNT(DISTINCT user_id)` undercounts because one user can have multiple sessions/carts in a day.

```sql
WITH deduped_add_to_cart AS (
  SELECT DISTINCT
    session_id,
    request_id,
    user_id
  FROM events
  WHERE event_name = 'add_to_cart'
),
sessions_today AS (
  SELECT DISTINCT session_id
  FROM events
  WHERE event_name = 'search'
)
SELECT
  COUNT(DISTINCT d.session_id) * 1.0 / COUNT(DISTINCT s.session_id) AS add_to_cart_rate
FROM sessions_today s
LEFT JOIN deduped_add_to_cart d ON s.session_id = d.session_id;
```

Say explicitly: numerator dedupes by `request_id` (idempotency key for the retry), denominator is session-level to match "rate per session" — matching numerator/denominator granularity is the actual thing being tested here.

---

### Question 3 — Users Ranked by Recent Purchases

> Retrieve users along with a count of products purchased, sorted by most recent transaction date. (Source: aggregator/Karat-style prep content) 🟡

```sql
SELECT
  u.user_id,
  COUNT(DISTINCT t.product_id) AS products_purchased,
  MAX(t.transaction_date) AS last_transaction_date
FROM users u
JOIN transactions t ON u.user_id = t.user_id
GROUP BY u.user_id
ORDER BY last_transaction_date DESC;
```

---

### What This Round Tests

- Fluency with window functions on large, partitioned data (not just basic joins/group-by)
- Whether you instinctively question data quality (dedup, retries, granularity mismatches) before trusting a metric — a directly relevant skill for a team whose core job is catalog/data-quality ML
- Ability to state assumptions about grain (what does one row represent?) before writing a query
