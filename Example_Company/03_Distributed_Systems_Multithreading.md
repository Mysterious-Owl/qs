# Distributed Systems & Multithreading

Referral tip: **distributed system design with multithreading.** 🔵 One low-confidence public source corroborates a multithreading-flavored question in a Example Company DS loop: "explain multithreading, then write code for synchronization using synchronized threads," plus semaphores/deadlock-prevention system design. 🔴 (source URL not fully resolved — treat as directional, not verbatim)

---

### Question 1 — Explain Multithreading and Write Synchronized Code

> Explain the concept of multithreading. Then write code that synchronizes access to a shared resource across threads.

**Answer — concepts**

- A process has one memory space; threads within it share that memory space (heap, globals) but each has its own stack and instruction pointer.
- Multithreading gives concurrency (interleaved progress) and, on multi-core hardware, true parallelism.
- The core hazard: a **race condition** — two threads read-modify-write shared state without coordination, and the interleaving determines the (wrong) result.
- Fix with mutual exclusion: locks/mutexes, semaphores, or higher-level constructs (atomic types, concurrent queues).

**Answer — code (Python, since this is a DS role)**

```python
import threading

class Counter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1

    @property
    def value(self):
        with self._lock:
            return self._value

counter = Counter()

def worker(n):
    for _ in range(n):
        counter.increment()

threads = [threading.Thread(target=worker, args=(10_000,)) for _ in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter.value)  # deterministically 80,000 with the lock; without it, a race
```

Worth mentioning proactively: Python's GIL means threads don't give CPU-bound parallelism in CPython — for CPU-bound ML workloads you'd reach for `multiprocessing`, `joblib`, or push the parallel work into a native library (NumPy/Torch already release the GIL internally for heavy ops). Threads *do* still help for I/O-bound work (API calls, DB reads) in a serving pipeline.

---

### Question 2 — Deadlock Prevention

> How would you design a system to avoid deadlocks when multiple threads/processes need several shared resources?

**Answer**

Four classic necessary conditions for deadlock (Coffman conditions) — breaking any one prevents it:

| Condition | How to break it |
|---|---|
| Mutual exclusion | Not always avoidable (some resources are inherently exclusive) — but reduce lock scope/duration |
| Hold and wait | Acquire all needed locks atomically up front, or don't hold a lock while waiting for another |
| No preemption | Allow forced release with rollback (e.g., transaction abort + retry) |
| Circular wait | **Most practical fix**: impose a global lock ordering — every thread acquires resources in the same fixed order |

```python
# Circular-wait prevention: always acquire locks in a fixed, consistent order
def transfer(account_a, account_b, amount):
    first, second = sorted([account_a, account_b], key=lambda a: a.id)
    with first.lock:
        with second.lock:
            account_a.balance -= amount
            account_b.balance += amount
```

Also worth naming: lock timeouts + retry-with-backoff as a pragmatic detection-and-recovery approach when strict ordering isn't feasible across a large codebase.

---

### Question 3 — Distributed System Design Framing (multithreading → distributed)

> Take the same producer/consumer, shared-state idea and scale it across machines — how does your answer change?

**Answer — bridge from single-process to distributed**

The single-process primitives (locks, semaphores) don't cross machine boundaries. Map each concept to its distributed equivalent:

| Single-process concept | Distributed equivalent |
|---|---|
| Mutex/lock | Distributed lock (Redis `SETNX`/Redlock, ZooKeeper/etcd lease) |
| Shared memory counter | Centralized counter service, or CRDT/eventually-consistent counter if exact real-time accuracy isn't required |
| Thread queue | Message queue / stream (Kafka, SQS) — decouples producers from consumers, survives consumer crashes |
| Thread pool | Worker fleet behind a queue, autoscaled on queue depth |
| Deadlock via lock ordering | Same principle, but now also worry about network partitions — a "lock holder" can crash mid-hold, so use **leases with TTL**, not indefinite locks |

For a catalog-scale system (millions of SKUs, many writers touching overlapping product records), the practical answer is usually: avoid distributed locks where possible (they don't scale and are a single point of contention), and prefer **partitioning by key** (e.g., shard by `product_id` so each shard's writes are single-writer) plus **idempotent, retryable writes** over trying to hand-roll distributed mutual exclusion.

---

### What This Round Tests

- Correct mental model of shared-memory concurrency (races, locks, deadlock) — this is table-stakes CS fundamentals even for a DS role
- Whether you know the practical limits of naive concurrency thinking once it's distributed across machines (network partitions, crash-while-holding-lock)
- Whether you reach for the right level of abstraction: don't hand-roll a distributed lock when a partitioning/queue design avoids needing one at all
