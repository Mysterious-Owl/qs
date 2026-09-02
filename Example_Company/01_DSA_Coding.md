# DSA / Coding Round

Referral tip: **2 easy-to-medium DSA questions.** 🔵 Research corroborates the format broadly: 2–3 LeetCode Easy/Medium problems, ~30–60 min, need to fully pass ~2 of 3 with test cases, often alongside live Python/Pandas data-manipulation work rather than pure algorithmic depth. 🟢

---

## Format notes

- This is a **Data Scientist** loop, not SWE — expect the coding bar to reward *correct, clean, readable* solutions over highly-optimized ones, plus comfort translating a DSA idea into Pandas/NumPy when relevant.
- Recurring topics across sources: arrays/hashmaps, string manipulation, two pointers, basic linked lists, BFS/DFS graph traversal, basic DP. 🟢
- Specific problems reported by name: **"Kth largest element in an unsorted array"** (Source: SQLPad.io Example Company Labs DS case study) 🟡, **"merge two sorted arrays"** and **"longest substring without repeating characters"** (Source: Prepfully) 🟡, and a **grid min-cost-path DP** ("start top-left, move only right/down, minimize total cost") (Source: Medium, single-author Medium series — single author, unverified) 🔴.

---

### Question 1 — Kth Largest Element in an Array

> Given an unsorted array of integers, find the k-th largest element.

**Answer approach**

Three viable solutions, worth stating the trade-off explicitly since that's what a DS interviewer is probing:

| Approach | Time | Space | When to use |
|---|---|---|---|
| Sort descending, index `k-1` | O(n log n) | O(1) extra | Simplest; fine if n is small or you need it once |
| Max-heap of size n, pop k times | O(n + k log n) | O(n) | Rarely better than sort unless k ≪ n |
| Min-heap of size k | O(n log k) | O(k) | Best when k ≪ n and array is large/streaming |
| Quickselect (partition-based) | O(n) average, O(n²) worst | O(1) | Best average-case if worst-case risk is acceptable |

```python
import heapq

def kth_largest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]
```

Say out loud: "If this ran once on a batch, I'd just sort. If it's called repeatedly on a stream (e.g., top-k fraud scores per minute), I'd keep a persistent min-heap of size k." — ties the DSA answer back to the kind of streaming/ranking problem this team actually has (fraud scores, catalog relevance scores).

---

### Question 2 — Longest Substring Without Repeating Characters

> Given a string, find the length of the longest substring without repeating characters.

**Answer approach: sliding window + hashmap, O(n)**

```python
def longest_unique_substring(s):
    last_seen = {}
    start = 0
    best = 0
    for i, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= start:
            start = last_seen[ch] + 1
        last_seen[ch] = i
        best = max(best, i - start + 1)
    return best
```

Walk through the invariant clearly: `start` is the left edge of the current window; when a repeat is found *inside* the current window, jump `start` past the previous occurrence rather than incrementing one at a time — that's what makes it O(n) instead of O(n²).

---

### Question 3 (lower confidence, still practice) — Min-Cost Grid Path

> Given an `m x n` grid of costs, find the minimum-cost path from top-left to bottom-right, moving only right or down.

**Answer approach: bottom-up DP, O(mn) time, O(mn) space (reducible to O(n))**

```python
def min_path_cost(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
```

Mention the space optimization (single rolling row, O(n) space) proactively — it's a natural follow-up.

---

### Question 4 — Sort a string and find its longest run

> "Sort & highest sequence in the string."

**First move: clarify.** This prompt is genuinely ambiguous, and asking which reading they mean is part of a good answer rather than a stall — "highest sequence" could mean at least three different things. Say so, then offer the interpretations:

1. Sort the characters, then find the **longest run of one repeated character**.
2. Sort, then find the **longest consecutive alphabetical sequence** (`a,b,c,d`).
3. Find the longest such run in the **original** string, no sorting.

Most commonly it's #1 or #2. Below covers both, sharing the sort step.

**Interpretation 1 — longest run of a repeated character after sorting**

Sorting groups identical characters together, so the answer becomes the largest character count — which means you don't actually need to sort at all:

```python
from collections import Counter

def longest_run_after_sort(s):
    if not s:
        return "", 0
    counts = Counter(s)
    ch, n = max(counts.items(), key=lambda kv: kv[1])
    return ch * n, n
```

Worth saying out loud: sorting is O(n log n), but since sorting only groups equal characters, a frequency count answers it in **O(n)**. Recognizing that the sort is unnecessary is the point of the question. If they explicitly want the sorted string produced too, `"".join(sorted(s))` and then scan runs:

```python
def sorted_string_and_longest_run(s):
    t = "".join(sorted(s))
    best_ch, best_len = "", 0
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] == t[i]:
            j += 1
        if j - i > best_len:
            best_ch, best_len = t[i], j - i
        i = j
    return t, best_ch * best_len
```

**Interpretation 2 — longest consecutive alphabetical sequence after sorting**

Deduplicate first (otherwise repeats break the chain), then scan for consecutive code points:

```python
def longest_consecutive_sequence(s):
    chars = sorted(set(s))
    best_start = best_len = 0
    cur_start = cur_len = 0
    for i, ch in enumerate(chars):
        if i > 0 and ord(ch) == ord(chars[i - 1]) + 1:
            cur_len += 1
        else:
            cur_start, cur_len = i, 1
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len
    return "".join(chars[best_start:best_start + best_len])
```

`"abcxyzabd"` → sorted unique `a,b,c,d,x,y,z` → longest consecutive run is `xyz` (3) vs `abcd` (4), so `abcd`.

Complexity: O(n log n) from the sort, O(k) for the scan. Mention that for a fixed alphabet (26 letters, or 128 ASCII) you can bucket-count in **O(n)** and skip comparison sorting entirely.

**Edge cases to state**: empty string, single character, all-identical characters, mixed case (does `A` count as consecutive with `a`? — ask), digits/spaces/punctuation, and whether ties should return the first occurrence.

---

### What This Round Tests

- Can you recognize standard patterns (sliding window, heap-for-top-k, grid DP) quickly without over-engineering
- Clean, bug-free Python under light time pressure
- Whether you can connect a generic DSA pattern to the kind of ranking/streaming/dedup problems this specific team actually solves (worth doing explicitly — it signals you understood *why* they ask this, not just that you memorized the pattern)
