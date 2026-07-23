# Decoding Strategies — Part 1

---

## What Is a Decoding Strategy?

After the language model produces a probability distribution over the vocabulary for the next token, the **decoding strategy** is the algorithm that decides which token to actually pick from that distribution.

```
Language Model → Logits → Softmax → Probabilities → Decoding Strategy → Selected Token
```

---

## Greedy Decoding

**Greedy Decoding** selects the highest-probability token at each generation step, without considering future consequences.

### The Core Problem: Local vs. Global Optimality

- **Local optimum**: choose the best word at the **current step only**. Greedy does exactly this.
- **Global optimum**: choose the sequence of words whose **overall** probability is highest.

Greedy Decoding can fail to produce the best overall sequence, because it's fundamentally **myopic (short-sighted)** — it optimizes one token at a time, rather than the entire sequence.

### Three Major Problems This Causes

1. **Repetition of phrases** — the model repeatedly generates the same word or phrase (e.g., *"I think that I think that..."*).

2. **Loops** — it gets trapped in a repeating sequence of multiple tokens (e.g., *"and bought some food and bought some food..."*).

3. **Local vs. Global Optimality** — the model chooses the best immediate token (local optimum) but misses a sequence with a higher overall probability (global optimum), because it never looks ahead or revisits earlier decisions.

> 💡 **Root cause of all three**: Greedy Decoding never reconsiders a choice once made, and never looks ahead — it commits fully to whatever looks best *right now*, even if that choice paints the model into a corner a few tokens later.

---

## Beam Search

Beam Search addresses greedy decoding's myopia by tracking multiple candidate sequences ("beams") simultaneously, rather than committing to a single best token at each step.

*Reference: [Complete worked example of Beam Search](https://chatgpt.com/s/t_6a6041506fc88191b8de2c64c55376a8)*

### Beam Search Variants

#### 1. Length Normalization

*Reference: [ChatGPT conversation on Length Normalization](https://chatgpt.com/s/t_6a6042ea6b8c81918ae61853246b7871)*

> 💡 **Why this is needed**: since sequence probability is a product of per-token probabilities (each less than 1), longer sequences naturally accumulate a lower total probability than shorter ones — purely due to length, not quality. Length normalization corrects for this bias so beam search doesn't unfairly favor short outputs.

#### 2. Diverse Beam Search

*Reference: [ChatGPT conversation on Diverse Beam Search](https://chatgpt.com/s/t_6a60457a99348191b71e5be3300e3cde)*

> 💡 **Hiring manager analogy**: imagine a hiring manager selecting 3 candidates for a job.
> - **Standard Beam Search**: hires 3 candidates with virtually identical resumes from the same university, because they have the highest test scores.
> - **Diverse Beam Search**: hires candidate #1 with the highest score, but then applies a **penalty** to anyone else with the exact same background — forcing candidates #2 and #3 to come from completely different backgrounds/experiences.

This is exactly the problem Diverse Beam Search solves for text generation: standard beam search's top candidates often look nearly identical (differing by one word), so a diversity penalty pushes the beams toward genuinely different outputs.

#### 3. Constrained Beam Search

**Constrained Beam Search** is a variant of Beam Search that generates the highest-probability output sequence **while satisfying one or more predefined constraints** — such as requiring specific words or phrases to appear in the generated text.

- Unlike standard Beam Search (which only optimizes for sequence probability), Constrained Beam Search **tracks which constraints have been satisfied** by each beam during decoding, and continues expanding candidate sequences accordingly.
- At the end of the search, it selects the highest-scoring sequence that satisfies **all** the specified constraints.

This makes it especially useful for machine translation, dialogue systems, and text generation where certain keywords, entities, or domain-specific terminology **must** be included in the final output.

---

## Why Deterministic Methods Fall Short for Creative Tasks

### Deterministic Methods Lack Variety

Greedy Search and Beam Search are both **deterministic**. Given the exact same prompt, they will produce the **exact same output every single time** — there's no element of randomness or variation.

### The Problem for Creative Tasks

For tasks like dialogue/chatbots, story generation, or creative writing, this determinism is a drawback.

- Always picking the highest-probability words produces **robotic, repetitive, and boring** responses that lack human-like flair.
- These tasks require diversity and surprise.

### The Solution: Sampling

Sampling-based methods introduce **controlled randomness** by picking words according to their probability distribution, rather than always picking the single top-ranked option.

- This lets the model produce different, creative, and novel responses each time you run the same prompt.

---

## Sampling-Based Strategies

### Pure Random Sampling

*Reference: [ChatGPT conversation on Pure Random Sampling](https://chatgpt.com/s/t_6a60495d31f0819196625cd6c6846c39)*

**What does "Probability Distribution" mean here?**

Before generating the next word, the model doesn't just guess a single token. Instead, it looks at its entire vocabulary (typically 30,000 to 100,000 tokens) and assigns a probability score to **every single token**. The collection of all these probabilities, added together (summing to 100%, or 1.0), is the **probability distribution**.

**The pipeline:**

1. **The Language Model produces Logits.**
2. **Softmax converts Logits into Probabilities.**
3. **The Decoding Strategy takes over** — Pure Random Sampling uses **all** these probabilities, randomly sampling a token according to them.

### Pure Random Sampling: Mathematical Formulation

*Reference: [ChatGPT conversation on the math formulation](https://chatgpt.com/s/t_6a604b027dcc819189ab41a9a8014e6a)*

```
y ~ Categorical(p_1, p_2, ..., p_V)
```

**In simple words:**
- The model computes a probability for every token using Softmax.
- These probabilities form a **categorical probability distribution**.
- One token is randomly selected — tokens with higher probabilities are more likely to be chosen, but **every** token with a non-zero probability has *some* chance of being selected.

### Core Takeaway: The Downside of Pure Random Sampling

Pure random sampling gives creative and varied outputs, but because it samples from the **entire** vocabulary, it occasionally pulls bad or contradictory words from the bottom of the probability list.

> 💡 **Example**: *"Bird sat on the heat"* — "heat" is totally irrelevant here, but it still had some non-zero probability of being sampled.

When generating a sequence of 20 or 50 tokens, you effectively "roll the probability wheel" at **every single step**. Over a multi-step sentence, the probability of picking at least one weird or contradictory token increases drastically.

> ⚠️ This exact problem led to the invention of **Top-k** and **Top-p (Nucleus) Sampling**, which chop off the low-probability tail before sampling — see below.

---

## Temperature Scaling

For general-purpose modern LLMs, **Temperature Sampling combined with Nucleus (Top-p) Sampling** is the most widely used strategy. This approach samples tokens randomly from a *restricted* probability distribution, balancing creative, human-like responses with factual coherence.

**Temperature scaling** dynamically balances **greediness** (determinism) and **creativity** (diversity/randomness) in generative AI models.

### Why Temperature Scaling Is Needed

Temperature Scaling controls the randomness of text generation by adjusting the model's **confidence** in its predicted probabilities, *before* token selection.

- **Lower temperatures** → more deterministic outputs.
- **Higher temperatures** → more diverse and creative outputs.

> 💡 **Why not just change the decoding strategy instead?** The probability distribution produced by the language model may be too **confident** (overly peaked) or too **uncertain** (too flat). Simply changing the decoding strategy (Greedy, Top-k, or Top-p) *cannot* adjust the model's confidence — it can only decide *how* to select tokens from the given probability distribution. Temperature Scaling is the tool that actually reshapes the distribution itself, before any selection strategy is applied.

### How It Works

Temperature Scaling modifies the **logits** *before* Softmax, making the probability distribution **sharper** (lower temperature) or **flatter** (higher temperature). This gives better control over the trade-off between deterministic/coherent text and diverse/creative text — letting the subsequent decoding strategy operate on a more suitable probability distribution.

**Motivation**: create a "knob" that controls how "sharp" or "flat" the distribution is before sampling.
- **High temperature** → more random (flatter distribution).
- **Low temperature** → more deterministic (sharper distribution).

*Reference: [ChatGPT conversation on temperature scaling motivation](https://chatgpt.com/s/t_6a6051031dfc819191911fa2a2fdebb3)*

```
Language Model
      │
      ▼
   Logits
      │
      ▼
Temperature Scaling (optional)
      │
      ▼
    Softmax
      │
      ▼
Probabilities
      │
      ▼
Decoding Strategy
(Greedy / Beam / Random / Top-k / Top-p)
```

*Reference: [How changing temperature T affects the Softmax distribution](https://chatgpt.com/s/t_6a6052872ee481918df801a38563d654)*

> ⚠️ Even with temperature scaling, there's still a decent chance of producing nonsense tokens — this is why temperature is typically **combined** with other techniques (like Top-p) rather than used alone.

### Works With Any Decoding Strategy

Temperature Scaling can be combined with Greedy Search, Random Sampling, Top-k Sampling, Top-p Sampling, or Beam Search, to adjust the model's confidence before token selection.

### Weaknesses of Temperature Scaling

- **Cannot generate text by itself** — it only modifies the probability distribution and must be combined with a decoding strategy.
- **Cannot eliminate** low-quality or irrelevant candidate words from the probability distribution (it reshapes the whole curve, but doesn't remove any options).
- **Task-dependent tuning**: selecting an appropriate temperature value is task-dependent and often requires experimentation.
- **Very high temperature** can produce incoherent or nonsensical text, by increasing the likelihood of low-probability tokens.
- **Very low temperature** can reduce output diversity, by making the model repeatedly select the highest-probability tokens.

---

## Top-k Sampling

*Reference: [ChatGPT conversation on Top-k Sampling](https://chatgpt.com/s/t_6a60570c6c6c819192bfa96b45df6fd2)*

Top-k Sampling restricts sampling to only the **k** highest-probability tokens, discarding the long low-probability tail before sampling.

### The Limitation: A Fixed `k` Can't Adapt

A fixed value of `k` cannot adapt to different contexts.

- Depending on what point you're at in a sentence, the number of "sensible" next words changes dramatically.
- Sometimes you only want **2** options; sometimes you need **30** options.

> ⚠️ This inflexibility — a static cutoff regardless of context — is exactly what motivates **Top-p (Nucleus) Sampling** below.

---

## Top-p (Nucleus) Sampling

*Reference: [ChatGPT conversation on Top-p (Nucleus) Sampling](https://chatgpt.com/s/t_6a6059939ff081919521c857fd781375)*

Instead of a fixed count of tokens (like Top-k), Top-p Sampling dynamically selects the **smallest set of tokens** whose cumulative probability exceeds a threshold `p` — adapting naturally to how "peaked" or "flat" the distribution is at each step.

*Reference: [How to compute Cumulative Probability](https://chatgpt.com/s/t_6a60e16f9dac81919d62e222381bc730)*

### Combining Temperature Scaling With Top-p Sampling

*Reference: [ChatGPT conversation on combining Temperature + Top-p](https://chatgpt.com/s/t_6a60e21b8e788191ab0700cdb957b9d6)*

> ⚠️ **A common misconception**: this decoding method does **not** make the model "more intelligent." It does not. It only changes **how** the next token is selected from the probabilities predicted by the language model. If the underlying model assigns high probability to an incorrect fact, Temperature Scaling + Top-p Sampling **cannot fix that** — it can only sample from the probabilities it's given. Decoding strategy and model knowledge are two entirely separate things.

---

## How Logits Are Computed

*Reference: [ChatGPT conversation on how logits are computed](https://chatgpt.com/s/t_6a60e2dd7c288191982c71338663fd22)*
