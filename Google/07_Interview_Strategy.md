# Interview Strategy — The Follow-Up Ladder

The most useful thing in the source post, and the reason to read it at all. Everything else is a topic list you could assemble yourself; this is the part about *how the rounds actually run*.

---

## The core observation 📄

> "The biggest difference between preparation and the actual interview is the **depth of follow-up**."

The candidate's own illustration:

```
          You think you've answered the question.
                        ↓
                     "Why?"
                        ↓
                  You answer.
                        ↓
              "What if X happens?"
                        ↓
                  You answer.
                        ↓
            "How would you scale it?"
                        ↓
                  You answer.
                        ↓
              "What's the tradeoff?"
                        ↓
   …three or four levels deeper than the original question.
```

And the conclusion that follows from it:

> "The interviewer isn't necessarily looking for a rehearsed perfect answer. **They're evaluating how you reason through unfamiliar problems.**"

## What this changes about preparation 🧩

If the round is a ladder rather than a question, then preparation that optimises for the *first* answer is optimising for the wrong thing. Three consequences:

**Depth beats breadth, past a point.** Knowing forty topics one level deep gets you through forty first answers and then fails at every rung two. Knowing fifteen topics four levels deep survives the ladder. The post's own topic lists are broad — but the candidate's *lesson* is about depth, and when those conflict, follow the lesson.

**You will be pushed past what you know.** That's the design, not a failure. The ladder only stops when you reach the edge of your knowledge, so *everyone* ends each thread at their limit. What's being scored is what you do there — reason from principles, state your uncertainty and what you'd check, or bluff. The first two pass.

**Rehearsed answers are actively risky.** A polished, fast first answer invites a harder second question, and the contrast between a fluent level one and an empty level two reads worse than a thoughtful level one would have.

## Practising the ladder 🧩

The post's advice under *what I'd do differently*:

> "**Practice follow-up questions.** Don't stop after giving the first answer. Ask yourself: *What would the interviewer challenge here?*"

Concretely: after any answer you can give fluently, generate four rungs against it. The four that recur most:

| Rung | The question behind it |
|---|---|
| **Why?** | Do you know the mechanism, or the conclusion? |
| **What if the assumption breaks?** | Do you know the preconditions? |
| **How does it scale?** | Does it survive 1000× data / QPS / users? |
| **What's the trade-off?** | What did you give up — and is there a case where you'd choose otherwise? |

Worked example on one answer 🧩:

> *"I'd use XGBoost."*
> — **Why?** Tabular data, non-linear interactions, strong default accuracy.
> — **What if the relationship is basically linear?** Then a regularized linear model matches it and extrapolates, which trees can't — their predictions go flat outside the training range.
> — **How does it scale?** Histogram splitting, feature subsampling, distributed training; but latency per tree grows with count, so distil or cap depth if there's a tight budget.
> — **Trade-off?** Lost interpretability and extrapolation, gained accuracy and tolerance of messy features. In a regulated setting I'd trade back — scorecard or monotonic-constrained model.

Four rungs, one claim. That's the unit of practice.

## The candidate's other lessons 📄

**What went well** — real production experience (trade-off discussions land differently when you've lived them); taking system design seriously rather than as an afterthought; revisiting fundamentals ("painful but extremely useful"); practising explaining concepts **out loud**; using their own projects for behavioural and architecture answers.

**What they'd do differently** — start coding prep sooner; don't neglect CS fundamentals; practise ML system design out loud; practise follow-ups; don't over-index on GenAI.

**The takeaways, verbatim:**

> → L5 isn't just about solving coding problems.
> → Strong ML fundamentals still matter even if you work in GenAI.
> → System design requires depth, not buzzwords.
> → Interviewers care about your reasoning and tradeoffs.
> → Communication is almost as important as the technical answer.
> → Real production experience is extremely valuable.
> → Don't assume that knowing something means you can explain it clearly under pressure.
> → Prepare for follow-up questions, not just the initial question.

## Two themes running underneath 🧩

**"Out loud" appears three separate times** — in what went well, in what they'd do differently, and in the takeaways. It's the most-repeated piece of advice in the post and the easiest to skip, because reading feels like progress and talking to an empty room feels stupid. The last takeaway is the sharpest statement of the gap: *knowing something and being able to explain it under pressure are different skills*, and only one of them is trained by reading.

**It was their third attempt** 📄, mentioned only in a comment reply. The post reads as a systematic-preparation story; the third-attempt detail reframes it as a *persistence* story with systematic preparation layered on. Both are true, and the second is easy to miss.

## A closing caveat 🧩

Per [the track README](README.md): this is one person's account, the outcome was still pending at the time of writing, and the loop shape was disputed in the comments. The strategy in this file — ladder your own answers, practise aloud, prefer depth — is good preparation advice regardless of whether every process detail in the post is exact. Take the method; hold the specifics loosely.
