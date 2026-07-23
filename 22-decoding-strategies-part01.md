# Decoding Strategies — Part 2

---

## Repetition Penalty

### The Problem

Transformer models naturally tend to get caught in repetitive loops during generation.

> 💡 **Example**: *"I like going to the park, the park, the park..."*

**Repetition Penalty** is another decoding technique used by many modern LLMs to reduce repeated words and phrases during text generation.

- **Its purpose is simple**: it decreases the probability of tokens that have already been generated, making the model less likely to repeat the same words or phrases.

**Examples of what it prevents:**
- *"The meeting was very very very very very important."*
- *"Thank you for your help. Thank you for your help. Thank you for your help."*

### Where Is It Applied in the Pipeline?

```
Input
   │
   ▼
Language Model
   │
   ▼
Raw Logits
   │
   ▼
Temperature Scaling
   │
   ▼
Repetition Penalty
   │
   ▼
Softmax
   │
   ▼
Top-p Sampling
   │
   ▼
Next Token
```

> 💡 **Notice**: Repetition Penalty is applied **before** Softmax, because it modifies the **logits** directly — just like Temperature Scaling. Both are logit-level adjustments that happen before the probability distribution is finalized, not after.

### Methods

*Reference: [ChatGPT conversation on repetition/presence/frequency penalty methods](https://chatgpt.com/s/t_6a60f723f2ec8191a6c1d89b8098a0ca)*

| Method | How it works |
|---|---|
| **Repetition Penalty** | Reduce the score of any previously generated token by **rescaling** its logit |
| **Presence Penalty** | Subtract a **fixed** penalty from any token that has appeared at least once |
| **Frequency Penalty** | Subtract a penalty **proportional to how many times** a token has already appeared — discouraging repeated use of the same token more and more the more it's used |

### Strengths & Weaknesses

**Strengths:**
- Directly addresses the repetition problem.
- Simple to implement, and tunable.

**Weaknesses:**
- Can **over-penalize** tokens that should legitimately repeat (like "the" or "is").
- Doesn't distinguish between **meaningful repetition** (saying "democracy" repeatedly in a political essay, where it's topically appropriate) and **degenerate repetition** (looping, where the model is just stuck).
- The penalties are **heuristic**, not principled — there's no deep theoretical justification for the exact penalty values, just empirical tuning.

---

## Min-p Sampling

*Reference: [ChatGPT conversation on Min-p Sampling](https://chatgpt.com/s/t_6a60f86a90108191856bdfb81ef97516)*

> 💡 **In relation to what came before**: like Top-k and Top-p, Min-p is another way of trimming the low-probability tail before sampling — but instead of a fixed count (`k`) or a fixed cumulative mass (`p`), it sets a probability floor **relative to the top token's probability**, so the cutoff automatically adapts to how confident the model is at each step.

---

## Contrastive Decoding

*Reference: [ChatGPT conversation on Contrastive Decoding](https://chatgpt.com/s/t_6a60fa4f27188191af47d42feb49e978)*

---

## Speculative Decoding

*Reference: [Introduction to Speculative Decoding](https://chatgpt.com/s/t_6a60ffc2c1048191b86b1e5dcd0f6791)*

*Reference: [Speculative Decoding — Detailed Algorithm](https://chatgpt.com/s/t_6a610634146881918b64b3f06908c715)*

*Reference: [Speculative Decoding — Complete Worked Example](https://chatgpt.com/s/t_6a610824b570819190297545d39c8ddd)*

> 💡 **Worth flagging**: unlike the other techniques in these notes (which are about *what* token to pick), Speculative Decoding is primarily a **speed/efficiency** technique — using a smaller, faster "draft" model to propose candidate tokens that a larger model then verifies, to speed up generation without changing the output distribution. It solves a different problem (latency) than repetition penalty or nucleus sampling (output quality/diversity).

---

## How Modern LLMs Actually Combine These Techniques in Practice

*Reference: [How modern language models actually perform decoding in practice, combining multiple techniques](https://chatgpt.com/s/t_6a610a6eb2808191a008ddfd292d93ef) ⭐ highly recommended*

> 💡 In production, these techniques are rarely used in isolation — a real system typically stacks several of them together in the pipeline shown above (Temperature Scaling → Repetition Penalty → Softmax → Top-p Sampling, for instance), each addressing a different failure mode simultaneously.

---

## Key Takeaway

The central challenge of decoding is converting the language model's probability distribution over possible next tokens into **one selected token**.

- Choosing only the highest-probability token produces **coherent but repetitive** text.
- Excessive randomness creates **diverse but often incoherent** text.

Therefore, modern decoding strategies are designed to operate **between these two extremes** — selecting tokens that are probable enough to preserve coherence, while remaining variable enough to produce natural, engaging, human-like language.

> This balance reflects an important property of human communication: **effective language is neither completely predictable nor completely random.**
