# Coding / Python

Three coding rounds in the reported loop 📄 — the largest single block, and the area the candidate says they spent the most time on despite 8+ years of ML experience.

> "Even experienced ML engineers can get rusty with interview-style coding." — one of the candidate's five *"what I'd do differently"* points 📄

---

## The topics named 📄

Listed verbatim in the post, grouped here by what they actually test:

| Group | Topics |
|---|---|
| Linear structures | Arrays, Strings, Hash maps |
| Two-pointer family | Two pointers, Sliding window |
| Search | Binary search |
| Non-linear | Trees, Graphs, Heaps |
| Recursion family | Recursion, Dynamic programming |
| Cross-cutting | Complexity analysis |

Nothing exotic. This is the standard interview canon, which is itself the signal: an L5 **ML** loop still expects fluent general-purpose problem solving, not ML-flavoured coding.

**On whether ML gets implemented from scratch** — a commenter asked exactly this ("does Google ask you to implement ML or DL algorithms from scratch, or is it purely leetcode?"). The post doesn't answer it, and the author didn't reply to that comment. So: **unknown**. Don't infer either way from this source.

## The actual lesson 📄

The candidate's emphasis is not on topic coverage but on delivery:

> "Don't just practice getting the answer. Practice explaining your thought process."

The structure they name:

```
Here's my approach  →  here's why it works  →  here's the complexity  →  here are the edge cases
```

And separately: they deliberately strengthened **Python itself**, on the reasoning that *"knowing an algorithm isn't enough if you're struggling with the language while implementing it."*

## What that means in practice 🧩

The four-part structure above is worth treating as a literal script, because each part fails differently:

- **Approach before code.** State the plan in one or two sentences and get a nod before typing. Silent typing forfeits the signal the interviewer is there to collect, and if the approach is wrong you've burned the round.
- **Why it works.** The invariant, in words. For sliding window: *what is true of the window at all times*. For DP: *what the state means and why the recurrence is valid*. This is where two candidates with identical code separate.
- **Complexity, unprompted.** Time and space, and say what dominates. If you're asked rather than volunteering, you've lost a small amount of credit.
- **Edge cases, named before they're found.** Empty input, single element, duplicates, overflow, all-identical values, cycles in a graph. Naming them yourself reads as engineering judgement; being shown them reads as carelessness.

**On Python specifically** 🧩 — the fluency that matters under time pressure is narrow and worth drilling: `collections.defaultdict` / `Counter` / `deque`, `heapq` (and the min-heap-only trick of negating for a max-heap), slicing, `enumerate`, `zip`, sorting with `key=`, and comfort with the fact that `dict` preserves insertion order. Fumbling `heapq` syntax mid-round costs more than not knowing the algorithm.

## Where the depth lives 🔗

This repo has worked solutions for the canonical patterns, with the same "explain the invariant" discipline:

- [`../Example_Company/01_DSA_Coding.md`](../Example_Company/01_DSA_Coding.md) — top-k with a heap, sliding window, grid DP, and a string problem, each with the trade-off table and the out-loud framing
- [`../ML_Fundamentals/ML_Training_Notebook.ipynb`](../ML_Fundamentals/ML_Training_Notebook.ipynb) — for the separate skill of writing *fluent* ML code live, which is not the same as DSA

## The follow-up ladder, applied here 🧩

Per [07](07_Interview_Strategy.md), expect the question not to end when the code runs:

- *"What's the complexity?"* → then *"can you do better?"*
- *"What if the input doesn't fit in memory?"* → streaming, external sort, or a bounded-size heap
- *"What if this were called a million times a second?"* → precompute, cache, amortize
- *"What if the input were sorted / nearly sorted / had duplicates?"* → often unlocks a better bound
- *"How would you test this?"* → the edge-case list, plus a property-based check

Having a second, better approach in reserve is worth more than a fast first answer.
