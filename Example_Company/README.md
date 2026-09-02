# Example Company — Data Scientist, Catalog Team

Prep materials for a Data Scientist interview on a **Catalog Data Science** team. This team owns ML for the product catalog: duplicate product detection, compliance/content-moderation (nudity, weapons/arms, prohibited items), and product attribute extraction + content generation (increasingly GenAI/LLM-driven).

## Confidence key

Every question below is tagged with where it came from and how sure I am it reflects reality:

| Tag | Meaning |
|-----|---------|
| 🟢 **High** | Corroborated across 3+ independent sources, or confirmed by Example Company's own published content |
| 🟡 **Medium** | Echoed in 2+ places, or a single strong/specific source |
| 🔴 **Low** | Single, unverified mention — still worth prepping since it's topically on-target, but don't over-index |
| 🔵 **Referral** | From a contact who interviewed on this team ~2023 — not independently verified, but the most team-specific signal available |

## Process overview 🟡

Reported loop (varies by source): Resume screen → Recruiter/HR phone screen → Online Assessment or technical phone screen → 1–2 Coding rounds (LeetCode Easy/Medium + Python/Pandas + SQL) → ML breadth-and-depth round → ML case study / system design round → Hiring Manager round. Total timeline ~3–8 weeks. Self-reported difficulty ~3/5.

(Source: Glassdoor Example Company DS interview page, InterviewQuery guide, 1point3acres threads — all via search-snippet summaries, direct fetch blocked by these sites)

## Topic index

| # | File | Covers |
|---|------|--------|
| 1 | [01_DSA_Coding.md](01_DSA_Coding.md) | 2 easy–medium DSA questions the referral mentioned, plus corroborated patterns |
| 2 | [02_SQL.md](02_SQL.md) | Heavily emphasized for this role — window functions, dedup reasoning |
| 3 | [03_Distributed_Systems_Multithreading.md](03_Distributed_Systems_Multithreading.md) | Distributed system design + multithreading round |
| 4 | [04_ML_System_Design_Fraud_Detection.md](04_ML_System_Design_Fraud_Detection.md) | Fraud detection system design (confirmed previously asked) |
| 5 | [05_Ecommerce_Catalog_System_Design.md](05_Ecommerce_Catalog_System_Design.md) | Ecommerce compliance system design + general catalog-service design |
| 6 | [06_Duplicate_Product_Detection.md](06_Duplicate_Product_Detection.md) | Catalog team use-case #1 |
| 7 | [07_Content_Compliance_Moderation.md](07_Content_Compliance_Moderation.md) | Catalog team use-case #2 (nudity/arms/prohibited content) |
| 8 | [08_Attribute_Extraction_Content_Generation.md](08_Attribute_Extraction_Content_Generation.md) | Catalog team use-case #3 — grounded in Example Company's own published architecture |
| 9 | [09_Classical_ML_Statistics.md](09_Classical_ML_Statistics.md) | High-confidence, near-certain fundamentals |
| 10 | [10_LLM_Finetuning_RAG_GenAI.md](10_LLM_Finetuning_RAG_GenAI.md) | LLM fine-tuning, RAG, GenAI system questions |
| 11 | [11_DL_Architectures.md](11_DL_Architectures.md) | DL model architecture questions |
| 12 | [12_MLOps_CICD_LLM_Hosting.md](12_MLOps_CICD_LLM_Hosting.md) | Deployment/CI-CD/LLM hosting (most hires lean MLE) |
| 13 | [13_Behavioral.md](13_Behavioral.md) | Behavioral / culture-fit, Example Company's stated values (no Amazon-style LPs) |

## Biggest gap found in public research

No corroborated public writeup of the **exact** "design a duplicate-detection system" or "design a nudity/weapons moderation system" interview prompt — these seem to stay internal to this team rather than leaking to Glassdoor/Blind/etc. File 6, 7, and 8 lean on **Example Company's own published engineering blog** describing how its Catalog Data Science / Trust & Safety org actually builds these systems today — treat that as the most reliable substitute for a leaked question, since it's very plausibly what the interviewer has in mind as a "good answer."

## Expect a project deep-dive

Per the referral tip, expect at least one question to pull a project straight off your CV
and turn it into a system-design or deep-dive discussion. Map your own strongest two
projects onto this loop's themes before you go in — a GenAI/RAG platform maps onto files
[08](08_Attribute_Extraction_Content_Generation.md) and [10](10_LLM_Finetuning_RAG_GenAI.md);
anything fraud- or risk-scoring-shaped maps onto [04](04_ML_System_Design_Fraud_Detection.md)
almost one-to-one. Lead with those whenever a question is open-ended.
