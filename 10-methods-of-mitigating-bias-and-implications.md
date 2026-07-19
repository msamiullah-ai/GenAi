# Debiasing Word Embeddings & The First Neural Language Model

---

## Debiasing Methods

*Reference: [ChatGPT conversation on post-hoc debiasing](https://chatgpt.com/s/t_6a5ad2df8b4c81919ea3f5aae775b08f)*

The general goal of debiasing: **surgically remove the gender information while leaving everything else intact.**

### Post-hoc Debiasing — Bolukbasi et al. (2016)

This method identifies a "gender direction" in the embedding space and neutralizes gender-neutral words (like "doctor" or "engineer") along that direction. The final step is **Equalization**:

**Step: Equalization**

After neutralizing gender-neutral words, the algorithm adjusts *legitimately* gendered word pairs (such as he/she, king/queen, father/mother) so they become perfect mirror images of each other around the gender-neutral space.

- Their gender information is preserved, but they're positioned symmetrically, so every debiased neutral word is **equally similar** to both words in the pair.

> 💡 **Example**: after equalization, "doctor" has the same cosine similarity to "he" and "she" — ensuring no unintended gender preference, while "he" and "she" themselves remain distinct as gendered words.

### Counterfactual Data Augmentation (CDA)

**Core idea**: Instead of removing bias *after* training (like Bolukbasi et al.), remove bias from the **training data before training**.

- The model learns from data — so if the data is balanced, the learned embeddings will also be more balanced.

**How it works**: for every sentence containing gendered words, create another sentence by swapping the genders.

> 💡 **Worked example**:
> - **Original**: "The doctor told his patient she needed rest."
> - **Augmented**: "The doctor told her patient he needed rest."
>
> The model trains on *both* sentences. Now "doctor" co-occurs equally with male and female words, so the learned embedding has no gender preference.

**Advantages:**
- Fixes bias at the source (training data).
- No need for post-hoc debiasing after training.
- Embeddings naturally learn less biased representations.

**Disadvantages:**
- Corpus size roughly doubles.
- Only works for biases that can be systematically swapped, such as gender — it's much harder to create counterfactual versions for racial, cultural, or socioeconomic biases.

### Learning Fair Representations

- A **fairness regularization term** is added to the loss function.
- The optimizer is penalized if it can predict gender from supposedly gender-neutral embeddings (e.g., occupations like doctor, engineer, teacher).
- As a result, the model learns embeddings that retain useful semantic information while making it difficult to infer protected attributes.

### Limitations of Debiasing

Debiasing is not a perfect solution:

- It requires **humans to decide** which attributes (gender, race, religion, nationality) and which groups should be protected — introducing another source of subjective bias.
- Methods like CDA may also **alter real-world data distributions**, potentially changing documented reality rather than merely removing unfairness.
- If a debiased model is later **fine-tuned on biased data**, it can relearn the same biases — meaning debiasing is not a one-time fix, and models must be continuously monitored and re-evaluated.

> ⚠️ **When you allow changing training data… you also allow changing the documented reality as we know it.** The one who controls the data, controls what the models output.

> 💡 A debiased pretrained model can absolutely be **rebiased during fine-tuning**. Because fine-tuning adjusts the model's weights on a new dataset, it's highly prone to inheriting, amplifying, or even actively learning new biases based on how the fine-tuning data is composed.
>
> **Learning at test time** means the model continues to adapt or update itself during inference (testing/deployment) using new incoming data, without being retrained from scratch — this is a separate concept from fine-tuning, but raises similar re-biasing risks.

---

## Gonen & Goldberg (2019): "Lipstick on a Pig"

Gonen & Goldberg argued that post-hoc debiasing (like Bolukbasi et al.'s method) only removes the **obvious, measurable** gender component — it does not eliminate the underlying bias.

- Bolukbasi's method removes the *projection* of a word onto the gender direction, so a neutral word like "doctor" becomes equally similar to "he" and "she" (direct bias removed).
- However, the **overall structure** of the embedding space remains unchanged.

> 💡 **Example**: "nurse" may still be surrounded by words like "caregiver," "midwife," and "receptionist" — all female-stereotyped. As a result, a simple classifier can still infer the gender association of "debiased" words purely from their neighboring words.

**Their conclusion**: bias is **distributed throughout** the embedding space, not concentrated in a single direction. So removing one gender axis is like putting **lipstick on a pig** — it hides the bias rather than truly removing it. Complete debiasing likely requires changing *how embeddings are trained*, not just modifying them afterward.

### Broader Implications for NLP

- Bias is not limited to Word2Vec or GloVe — it propagates through **all** downstream models that use biased embeddings, including modern transformers like BERT and GPT, since they're trained on biased text.
- Standard NLP benchmarks measure **accuracy, not fairness** — so a model can perform well while still encoding harmful stereotypes.
- Current debiasing methods **reduce** bias but don't **eliminate** it completely — making bias an ongoing challenge requiring better training data, bias-aware evaluation, and responsible model design, not just technical post-processing.

---

## How Are Word Embeddings Formed? (Recap)

Word embeddings are dense numerical vectors learned automatically from large text corpora, based on the **Distributional Hypothesis**: words that appear in similar contexts tend to have similar meanings.

**The learning process:**

1. Collect a large corpus of text (books, Wikipedia, news, etc.).

2. Choose a context window (e.g., 2 words before and after the target word).

3. Train a model (Word2Vec, GloVe, or FastText) to learn relationships between a word and its surrounding context.

4. During training, the model adjusts each word's vector so that words occurring in similar contexts end up with similar embeddings.

> 💡 **Example**: *"The doctor treated the patient"* and *"The physician examined the patient"* provide similar contexts for "doctor" and "physician," so their vectors become close in the embedding space.

The final embedding is simply a **vector of learned numbers**. Word2Vec, GloVe, and FastText all produce **static** embeddings.

### Static vs. Contextual Embeddings

| Type | Meaning | Examples |
|---|---|---|
| **Static** | A word has exactly **one fixed vector**, regardless of the sentence it appears in | Word2Vec, GloVe, FastText |
| **Contextual (dynamic)** | The embedding **changes** depending on the sentence | ELMo, BERT, GPT |

After training Word2Vec, GloVe, or FastText, we get an **embedding matrix**. The embedding dimension (length of each embedding vector) is chosen **before** training — it's a hyperparameter set by the researcher or engineer, not something learned automatically.

---

## Course Recap: Key Stages Covered So Far

1. **Tokenization**: taking text and breaking it down into individual tokens.

2. **Representation**: tokens needed to be represented in a format a machine learning model could consume. One-hot vectors were deemed insufficient because they lack the geometric relationships necessary to capture meaning.

3. **Distributed Representations**: to solve this, we adopted the distributional hypothesis — words with similar meanings co-occur or are related — leading to the study of word embeddings (Word2Vec, Skip-gram, GloVe, CBOW).

4. **Critical Analysis**: alongside embeddings, we discussed the presence of societal biases within these models — acknowledging that while embeddings are useful for capturing *distributional meaning*, they also capture *distributional prejudices*.

5. **Moving to Neural Networks**: having established how to represent words in a form consumable by a neural network, we've reached the point where we can begin designing the neural networks themselves — starting with simple feed-forward neural networks.

---

## The Goal of a Language Model

> The goal of a language model is to **predict a word in a sequence**.

More precisely: the goal of a language model is to predict a **probability distribution**, in which the correct word that should be predicted has the **highest probability**.

---

## Making a Feed-Forward Neural Network (FFNN) Order-Preserving

A normal feed-forward neural network does **not** naturally understand sequence order. So we explicitly preserve order by keeping embeddings in their original positions.

**The approach**: convert one-hot word inputs into dense embeddings using a **shared embedding matrix**, preserve the order of those embeddings, and feed them into a neural network to predict the next word.

```
3 input words
      │
      ▼
One-hot representation
      │
      ▼
Shared Embedding Matrix
      │
      ▼
3 word embeddings
      │
      ▼
Concatenate embeddings
      │
      ▼
Feed Forward Neural Network
      │
      ▼
Softmax
      │
      ▼
Probability of every vocabulary word
```

### Worked Diagram: FFNN Language Model with Pre-trained Word2Vec Embeddings (Bengio-style Architecture)

*Predicting the next word from 3 context words.*

<img width="1496" height="707" alt="image" src="https://github.com/user-attachments/assets/a6765340-77dd-476b-87aa-779c1104f08f" />


**Walking through the architecture, layer by layer:**

1. **Input Layer** — 3 word indices. Example: `w_{t-3}` = "I", `w_{t-2}` = "like", `w_{t-1}` = "deep".

2. **Embedding Layer** — a row lookup from the pre-trained Word2Vec embedding matrix `W` (dimensions `d × |V|`, e.g. `300 × 10,000`). This matrix can be **frozen** or **fine-tuned** during training. Each of the 3 input words is looked up independently, producing three `300 × 1` embedding vectors (`e_{t-3}`, `e_{t-2}`, `e_{t-1}`). **No activation function** is applied here — it's a pure lookup, not a computational layer.

3. **Concatenate** — the three embedding vectors are **concatenated** (not averaged) into a single `900 × 1` vector: `[e_{t-3} ; e_{t-2} ; e_{t-1}]`.

   > 💡 **Why concatenate instead of average (unlike CBOW)?** Concatenation **preserves positional information** — the model can tell *which* word came first, second, and third. Averaging (as CBOW does) would throw that ordering away.

4. **Hidden Layer** — this is where an actual computational neural network layer appears (unlike the embedding layer, which was just a lookup). It applies a **non-linear activation** (`tanh` or `ReLU`):
   ```
   h = tanh(W₁x + b₁)
   ```
   With `W₁` sized `(512 × 900) + b₁`, producing a `512 × 1` output.

5. **Output Layer** — a **Softmax** over the entire vocabulary `|V|`, predicting the probability of the next word:
   ```
   P = softmax(W₂h + b₂)
   ```
   With `W₂` sized `(|V| × 512) + b₂`, producing a `|V| × 1` (e.g. `10,000 × 1`) probability distribution.

6. **Prediction** — take the `argmax` of that probability distribution to get the predicted word `w_t`.

   > **Example**: Given the input *"I like deep ___"*, the model predicts **"learning."**

### Further Reading

- [Complete explanation of the FFNN diagram above](https://chatgpt.com/s/t_6a5af2c8daf8819189917f5fdb920321) ⭐
- [Full explanation of everything covered so far — highly recommended](https://chatgpt.com/s/t_6a5af6b7ef208191ab1a6fdbb48c8889) ⭐⭐⭐⭐
