# Semantic Properties of Embeddings

---

## The Big Picture

Think of every embedding as a **recipe made of many hidden semantic ingredients**.

- Similar words have similar recipes.
- Adding vectors mixes recipes.
- Subtracting vectors removes ingredients.
- The resulting vector often corresponds to another meaningful word, because the embedding space is organized according to semantic relationships.

### The Formal Definition

Every word is represented by a point (vector) in a high-dimensional space. If the embedding size is 300, every word is a vector of 300 numbers.

Word embeddings represent each word as a vector in a high-dimensional space (typically 100–1000 dimensions), where words with similar meanings are located close to one another. An **embedding function** `E: V → R^d` maps every word in the vocabulary `V` to a `d`-dimensional real-valued vector.

> 💡 Although the individual dimensions are learned automatically and don't explicitly correspond to human-readable concepts like "royalty" or "masculinity," together they capture **latent semantic and syntactic properties** of words. Because these vectors encode meaning geometrically, simple vector operations become meaningful.

*Reference: [ChatGPT conversation on embedding space and precision loss](https://chatgpt.com/s/t_6a5a0771a5208191a3fdf6195e5d2fe7)*

> 💡 **A subtle limitation worth flagging**: models work with meaning in a continuous space where there is often meaning *without* a corresponding word. To turn a resulting vector back into something readable, the model has to "downsample" — find the closest actual word — which necessarily loses some precision of the original meaning.

---

## Vector Arithmetic: Addition vs. Subtraction

### What Does `E(king) − E(man)` Represent?

It represents the **transformation** (or relationship) — the **displacement** (direction and magnitude) — required to move from the embedding of "man" to the embedding of "king." It is **not** a word embedding itself; it's a vector representing the semantic *relationship* between the two words.

### What Happens When You Compute `E(man) + E(royal)`?

It produces a new vector whose position reflects the **combined semantic information** of "man" and "royal." It combines the semantic properties encoded in both embeddings into a single vector. If the embedding space has been learned well, this new vector often lies close to the embedding of a word that naturally combines those meanings.

### Comparing Addition vs. Subtraction

| Operation | What it produces |
|---|---|
| **Subtraction**: `E(king) − E(man)` | A relationship (displacement) vector — tells you how to move from "man" to "king" |
| **Addition**: `E(king) + E(man)` | A combined semantic vector — merges the semantic information of both words. If the combination corresponds to an existing concept, the result lies close to that word's embedding |

### The Full Analogy Derivation

1. Compute `r = E(king) − E(man)` → a **relationship (displacement) vector**, telling us how to move from "man" to "king" in the `n`-dimensional embedding space.

2. Compute `E(woman) + r` → we're applying that same relationship/displacement to the embedding of "woman." This produces a new vector representing "the embedding of a word that has the semantic properties of *woman*, transformed by the same relationship that transformed *man* into *king*."

3. Since embeddings are organized semantically, this new vector is expected to lie closest to the embedding of **"queen"** — not "king."

```
E(woman) + (E(king) − E(man)) ≈ E(queen)
```

> 💡 **What "embeddings are organized semantically" means**: during training, the model automatically arranges word embeddings in vector space so words with similar meanings or similar *relationships* occupy nearby or systematically related positions. A word's position in the embedding space isn't random — it reflects its meaning.

---

## Scaling an Embedding Vector

Scaling an embedding vector changes the **strength (weight)** of the semantic information it encodes, without changing its **semantic direction**.

- Scaling multiplies the embedding's magnitude, but does **not** by itself change the embedding's semantic meaning when similarity is measured using **cosine similarity** — because cosine depends only on direction.
- Conceptually, scaling doesn't create a new meaning or a "more royal" word. Instead, it adjusts **how much** that embedding's semantic contribution affects a resulting combined vector.

> 💡 **Best way to think about it**: scaling is a way of *weighting the importance* of an embedding's semantic features during vector composition — not a way of altering what it means.

---

## Consistent Relationships as Vector Offsets

Word embeddings don't only capture word meanings — they also capture **consistent semantic and syntactic relationships as vector offsets**.

Whether the relationship is gender, plurality, verb tense, adjective comparison, or geography, the **same vector offset** learned from one pair of words can often be applied to another word to obtain the corresponding related word. This consistency is what makes vector arithmetic and analogy-solving possible.

**Example — Plurality:**
```
apple → apples    +    (cars − car)
```

---

## The "Muddy Smoothie" Problem

In models like Word2Vec, the word "bank" does **not** have separate vectors for its different meanings (financial institution vs. river bank). Instead, the model assigns it a **single vector** that blends all those various "flavors" together.

> 💡 This is described as creating a **"muddy smoothie"** of meanings — a major limitation, because the model fails to distinguish between the distinct senses of a word based on context.

### Major Limitations of Static Embeddings (Word2Vec, GloVe, FastText)

*Reference: [ChatGPT conversation on static embedding limitations](https://chatgpt.com/s/t_6a5a3e22e6548191b4e6c996a55d224b)*

Static embeddings assign **one fixed vector per word**, regardless of context — which leads to several limitations:

1. **Polysemy & homonymy**: cannot distinguish multiple meanings of the same word — "bank" gets one embedding that blends all its senses.

2. **Non-compositional phrases and idioms**: struggle with phrases whose meaning isn't simply the sum of their individual words (e.g., "hot dog," "kick the bucket").

3. **Antonym confusion**: because these models learn from *contextual* similarity, antonyms such as "hot" and "cold" often get similar embeddings, since they appear in similar contexts.

4. **Poor negation modeling**: simply adding the embedding of "not" does not reliably transform the meaning of a word or sentence.

> 💡 **Root cause behind all four**: static embeddings are **context-independent** — the same representation gets assigned to every occurrence of a word, no matter how it's actually being used in a sentence. (This is exactly the gap that contextual embeddings like BERT were later designed to close.)

---

## Evaluating and Inspecting Word Embeddings

*Reference: [ChatGPT conversation on evaluating word embeddings](https://chatgpt.com/s/t_6a5a422cb14081919754dbb3d719367a)*

*Reference: [How t-SNE works](https://chatgpt.com/s/t_6a5a42851eb48191903eef005be80c88)*

### Downstream Tasks

In NLP, a **downstream task** is the specific, end-user application you actually want to solve. These are supervised problems that use a base, pre-trained model (like an LLM or BERT) as their foundation — instead of building a model from scratch, you fine-tune it or add a task-specific layer on top of the pre-trained model to achieve high performance on your target application.

### Extrinsic Evaluation

**Extrinsic evaluation** measures the quality of word embeddings **indirectly**, by evaluating how much they improve performance on a downstream NLP task.

- Instead of assessing the embeddings themselves in isolation, they're used as input features to an NLP model, and the model's performance (accuracy, F1-score, BLEU score, etc.) indicates the usefulness of the embeddings.
- If embeddings lead to better results on tasks like Named Entity Recognition (NER), Sentiment Analysis, Machine Translation, or Question Answering, they're considered high-quality embeddings.

---

## Embedding Space Geometry

### Ambient vs. Intrinsic Dimensionality

- **Ambient dimensionality**: the number of dimensions of the embedding space — i.e., how many numbers every embedding vector is represented with. All word embeddings within a given model/space share the same ambient dimensionality.

- **Intrinsic dimensionality**: the *minimum* number of dimensions actually needed to represent the underlying structure or semantic information in the data. It's not about how many dimensions the embeddings currently *have* — it's about how many dimensions the data actually *needs*.

### What Happens When Ambient >> Intrinsic Dimensionality

If the ambient dimensionality is much larger than the intrinsic dimensionality:

- **Upside**: the embedding space gains greater flexibility to organize and separate semantic concepts, reducing the need to compress different meanings into the same region.

- **Downside**: in very high-dimensional spaces, many word vectors can exhibit **artificially high cosine similarities**, making unrelated words appear more similar than they truly are. As a result, cosine similarity becomes **less discriminative** — reducing its ability to reliably distinguish genuinely related words from unrelated ones.

### Isotropy and Anisotropy

*Reference: [ChatGPT conversation on isotropy and anisotropy](https://chatgpt.com/s/t_6a5a4a0a5ee88191bd5cbb2b053dbc35)*

> 💡 **In simple words**: an *isotropic* embedding space spreads word vectors roughly evenly in all directions, while an *anisotropic* space has vectors clustered in a narrow cone — which is part of why cosine similarities can end up artificially inflated in practice (tying back to the dimensionality issue above).

### Semantic Drift — A Research Idea

> **Paper idea**: What will a word's meaning be 10 years from today?
>
> Example: "Apple" was just a fruit before the Apple iPhone was launched — word meanings shift over time as culture and technology evolve, and embeddings trained at one point in time can quickly become outdated reflections of a word's dominant sense.

---

## Societal Bias in Embeddings

> **Faithful embeddings = biased embeddings.**

Word embeddings do not understand what is true, fair, or ethical — they only learn the **statistical patterns** present in the training text. So if human language contains societal stereotypes or prejudices, the embeddings will encode those patterns as **semantic relationships in the vector space**.

**Definition**: Societal bias in word embeddings is the phenomenon where embeddings faithfully encode the statistical biases present in human language — causing stereotypes and prejudices in the training corpus to become **geometric relationships** in the embedding space, which can then influence downstream AI systems.

### Quantifying Bias: The Word Embedding Association Test (WEAT)

WEAT measures whether one group of words is systematically **closer** in the embedding space to one concept than another concept — a method for measuring bias in word embeddings.

- Instead of measuring human reaction time (as in the psychological test it's modeled after), WEAT measures **distance (cosine similarity)** between vectors.
- **Underlying assumption**: words that are closer together in embedding space are more strongly associated with each other.

*Reference: [Mathematical formulation of WEAT](https://chatgpt.com/s/t_6a5a7f09fb0081918ac0d9b60198507e)*
