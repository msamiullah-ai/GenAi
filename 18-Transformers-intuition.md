# The Transformer Architecture: Intuition to Full Encoder-Decoder Flow

---

## Reference Materials

*Slides: [Full lecture slide deck](https://pern-my.sharepoint.com/:b:/g/personal/namoos_qasmi_lums_edu_pk/IQC3sg4wLAEiR5a3Ly5e6yECASBDJOWz2Qqq8Mj3xsznuyM?e=Y2VyAu)*

---

## Why Transformers? The Core Motivation

The Transformer was introduced to overcome the biggest limitation of RNNs, LSTMs, and GRUs: **sequential processing**.

- In recurrent models, each word must wait for the previous word, because information is passed through the hidden state (`h_{t-1}`) — making training slow and difficult to parallelize.
- The Transformer removes this recurrence **entirely**, allowing all words in a sentence to be processed simultaneously.

To do this successfully, the encoder must still know:
- **What** each word is → using **word embeddings**.
- **Where** each word appears → using **positional encoding**.
- **Which** other words are most relevant for understanding it → using **self-attention**.

> 💡 **In simple terms**: every word asks three questions: *"Who am I?"* (embedding), *"Where am I?"* (position), and *"Who should I pay attention to?"* (self-attention).

### The Decoder Side

The decoder also abandons the RNN's hidden state and context vector, relying instead on attention mechanisms to generate output.

- Rather than compressing the entire sentence into a single "context vector" or "word smoothie," every input word keeps its **own** representation while communicating directly with every other word through self-attention — regardless of distance in the sentence.
- This captures long-range dependencies more effectively, while being trained efficiently on GPUs through parallel computation.

**In short**: the Transformer replaces recurrence with **self-attention + positional encoding**, achieving both richer contextual understanding and much faster training than previous sequence models.

> 💡 **Decoder flow of information is forward, not backward** — a decoder only ever builds on what's already been generated, never looking ahead.

---

## Core Idea of Transformers (Summary)

Instead of reading words one by one like an RNN, a Transformer processes **all words at once**, and lets each word directly determine which other words are most important, using self-attention.

**In 4 steps:**

1. Convert each token into an embedding.
2. Compute Query (`Q`), Key (`K`), and Value (`V`) for every token.
3. Each token compares its Query with the Keys of all tokens, to compute attention scores.
4. Use these scores to take a weighted combination of the Values, producing a context-aware representation for each token.

**Why is this powerful?**
- Every token can directly access information from every other token.
- All tokens are processed in **parallel**.
- Long-range relationships are captured effectively.

> **One-line intuition**: A Transformer lets every word "look at" every other word, decide what matters most, and update its meaning accordingly — all at the same time.

---

## Embeddings & Weight Tying

*Reference: [Cross-Attention in Transformer](https://chatgpt.com/s/t_6a5ee23eeddc8191bec1d351883cb952)*

*Reference: [Looking up the embeddings](https://chatgpt.com/s/t_6a5ee5349b3081919bc31f6d898005cc)*

> 💡 **Weight tying**: the embedding matrix used at the encoder input layer is **shared** with the one used at the projection layer at the final output layer — the same learned weights do double duty, reducing total parameters and often improving performance.

**The embedding dimension** is simply the length (number of elements) in the vector used to represent each word. If you choose 3-dimensional (3D) embeddings, every word is represented by a vector containing exactly 3 numbers.

---

## Positional Encoding

Transformers don't read words one-by-one like RNNs do, so they have **no natural sense** of "first word," "second word," or "100th word."

**The fix**: give each word a **positional ticket** made of multiple repeating functions — specifically, sine and cosine waves of different frequencies.

- **High-frequency** sine waves help the model tell adjacent words apart (short-term distance).
- **Low-frequency** sine waves help the model tell words far apart in a long paragraph (long-term distance).

*Reference: [How Transformers create Positional Encodings using sine and cosine functions](https://chatgpt.com/s/t_6a5eef965b40819193af5026fb64b7a6)*

Positional encoding adds sequence order information to token embeddings. Because Transformers use parallel attention layers rather than sequential processing, they cannot inherently track word order — this is the structural gap positional encoding fills.

### One Possibility: Alternating Sines and Cosines

<img width="1206" height="722" alt="image" src="https://github.com/user-attachments/assets/6c9a329e-ffbe-4571-af03-20822db5c5f3" />


- Use **alternating sines and cosines**, with **decreasing frequencies** across dimensions.
- This connects to **Euler's Identity**: `e^{jωt} = cos(ωt) + j·sin(ωt)`, where `j = √-1`.
- As shown in the diagram: the 1st dimension oscillates fastest, and each subsequent dimension (2nd, 3rd, 4th...) oscillates progressively slower — together forming a unique "fingerprint" per word position.

*(Source: "Attention for Neural Networks, Clearly Explained!!!" — [youtube.com/watch?v=PSs6nxngL6k](https://www.youtube.com/watch?v=PSs6nxngL6k))*

### The Formula

*Reference: [Sinusoidal positional encoding formulas from the original Transformer paper](https://chatgpt.com/s/t_6a5ef00e1e2c8191bf5e4074cfce151d)*

*Reference: [Clear summary](https://chatgpt.com/s/t_6a5ef18d1ea88191bd5f5c788878e10b)*

The positional encoding formula gives every token position (`0, 1, 2, 3, ...`) a unique `d_model`-dimensional vector, by filling the embedding dimensions with pairs of sine and cosine values.

For each pair of dimensions `(2i, 2i+1)`, it computes:
```
sin(pos / 10000^(2i/d_model))
cos(pos / 10000^(2i/d_model))
```

**Why the denominator matters**: the term `10000^(2i/d_model)` makes each dimension use a different wavelength (frequency):

- When `i` is **small**, the denominator is small, so the wave changes **rapidly** between neighboring positions — capturing fine, local position differences.
- When `i` is **large**, the denominator becomes very large, so the wave changes **slowly** — capturing long-range position information.

Together, these fast and slow waves create a unique positional "fingerprint" for every token, which is **added** to its word embedding — so the Transformer knows both *what* the word is and *where* it appears.

> 💡 **Why sinusoidal, specifically?** This design was chosen because it provides smooth, bounded values (between -1 and 1), works for arbitrarily long sequences (no need to retrain for longer inputs), and its mathematical properties let the model easily learn *relative* distances between words, not just absolute positions.

### Core Concepts Recap

- We choose the embedding dimension up front (e.g., `d_model = 512`), so every word embedding has 512 dimensions.
- For every word, we also need a 512-dimensional **positional encoding vector**, so position information can be added to every dimension of the embedding.
- For each pair of dimensions, the **even** dimension (`2i`) uses **sine**, and the **odd** dimension (`2i+1`) uses **cosine**.
- For small `i` (lower dimensions): the denominator is very small, so we compute approximately `sin(pos)` or `cos(pos)` — producing high-frequency waves. Nearby word positions (e.g., positions 5 and 6) get noticeably different values, making local positions easy to distinguish.
- For large `i` (higher dimensions): the denominator grows exponentially, producing low-frequency waves that change very slowly — capturing long-range positional information.
- Finally, this positional encoding vector is **added element-wise** to the word embedding vector, so the resulting representation contains both **what the word means** (embedding) and **where the word appears** (position).

### Visualizing the Positional Encoding Matrix

- Each **row** represents the positional encoding vector for one token position (Position 0, Position 1, Position 2, ...).
- Each **column** represents one dimension of that vector.
- Light and dark colors show the values produced by the sine/cosine formulas.

- **On the left** (lower dimensions, small `i`): the pattern changes very rapidly row-to-row, since these dimensions use high-frequency waves — making nearby word positions easy to distinguish.
- **Toward the right** (higher dimensions, large `i`): the denominator grows much larger, so waves change very slowly — many nearby positions have similar values here, capturing long-range positional information instead.

> 💡 **The key idea**: no single dimension uniquely identifies a position. Instead, the *combination* of all these fast-changing and slow-changing dimensions creates a unique positional fingerprint for every word.

---

## Self-Attention

**Self-attention** is the core mechanism in Transformer models that allows words in a sentence to interact with and influence each other. It calculates how much "attention" or weight every word should give to all other words in the same sentence — giving the model the ability to understand context and complex relationships.

### How the Transformer Builds a Context-Aware Representation

*Reference: [How the Transformer builds a context-aware representation for every word](https://chatgpt.com/s/t_6a5f08df6b5c8191b8fa241dd743dffd)*

Unlike an RNN, which processes one word after another, Self-Attention processes **all words simultaneously (in parallel)**.

**To compute the new representation for a particular word** (e.g., "the"):

1. The model first compares "the" with **every** word in the sentence, including itself, by computing a **dot product** (similarity score).
2. These raw scores are passed through **Softmax**, converting them into attention weights (normalized importance values that sum to 1).
3. Each word's embedding vector is multiplied by its corresponding attention weight — more relevant words contribute more, less relevant words contribute less.
4. All these weighted vectors are **summed** together to produce a new embedding for "the."

This new embedding no longer represents only the word "the" — it represents "the" **together with the context** provided by the entire sentence. The exact same four steps run for every word at the same time, letting the Transformer generate contextual embeddings for the whole sentence in parallel.

### Self-Attention via Query, Key, Value

*Reference: [How Self-Attention is computed using Q, K, V](https://chatgpt.com/s/t_6a5f09cd2c3081919c37477f863768aa)*

<img width="1366" height="762" alt="image" src="https://github.com/user-attachments/assets/9a723f2e-bba4-459e-8cd7-f9d59b3c6e93" />


### Complete Flow of Self-Attention Vector Formation

*Reference: [Complete flow, fully worked example](https://chatgpt.com/s/t_6a5f13abc8248191ad070485db0459aa) ⭐ recommended*

Walking through it for the word **"do"**:

1. The one-hot vector for "do" is passed to the **embedding layer**, converting it into an embedding vector.
2. This embedding vector is **added** to the positional encoding vector, producing the input representation of the word.
3. This representation is multiplied by three **learned weight matrices** `W_Q`, `W_K`, `W_V`, generating the **Query**, **Key**, and **Value** vectors.
4. The Query vector of "do" is compared with the **Key vectors of all words** in the sequence, using the dot product to compute similarity scores.
5. These similarity scores pass through **Softmax**, producing normalized attention weights.
6. Each attention weight is multiplied by its corresponding **Value** vector, producing weighted Value vectors.
7. All weighted Value vectors are **summed** to produce a single context vector (attention output) for "do" — now containing information from the entire sequence, according to the learned attention weights.

### Why Is the Query Multiplied With the Key Vectors?

*Reference: [ChatGPT conversation on this exact question](https://chatgpt.com/s/t_6a5f895ff4d0819196527568ba38c3d7) ⭐ recommended*

---

## Multi-Head Attention

### Why a Single Self-Attention Computation Isn't Enough

**1. Self-Attention learns relationships between words.** As established above, Self-Attention lets every word compare itself with every other word.

> 💡 **Example**: in *"The animal didn't cross the street because it was tired,"* the word "it" may attend strongly to "animal," while "tired" may also attend to "animal." The attention weights (`α`) learned by self-attention capture relationships/dependencies between words — instead of viewing words independently, the model learns which words are most relevant to one another.

**2. What is an Attention Head?**

Using the three matrices generated from the input representations — Query (`Q`), Key (`K`), Value (`V`) — the model performs **one complete Self-Attention computation**:

1. Compute similarity scores using `QK^T`.
2. Apply Softmax to obtain attention weights.
3. Use those weights to compute a weighted sum of the Value (`V`) vectors.

This **complete pipeline** — from generating Q, K, V to producing the final contextual representations — is called **one Attention Head**.

> An Attention Head is simply one independent Self-Attention mechanism with its own learnable Query, Key, and Value matrices.

**3. Why isn't one Attention Head enough?**

A real sentence usually contains many different kinds of relationships **at the same time**.

> 💡 **Example**: *"The doctor who treated the patient prescribed medicine because she was concerned."*
> - "she" should refer to "doctor" (coreference relationship).
> - "treated" connects doctor and patient (subject-object relationship).
> - "prescribed" relates to medicine (action-object relationship).
> - Nearby words may also form grammatical relationships, like adjectives modifying nouns.

A **single** attention head has limited capacity — it produces one set of attention weights, meaning it can mainly focus on **one** pattern of relationships at a time.

### Multi-Head Attention: The Fix

Instead of using only one Attention Head, Transformers use **multiple independent Attention Heads**.

- Each head has its own separate learnable matrices: `W_Q`, `W_K`, `W_V`.
- Because these matrices are different, every head learns to focus on **different aspects** of the sentence.

**Example of specialization:**
- Head 1 may learn grammatical relationships.
- Head 2 may learn subject-verb dependencies.
- Head 3 may learn long-distance references (pronouns referring to nouns).
- Head 4 may learn semantic similarity between related words.

All heads process the same input sentence in **parallel**, but each produces a different contextual representation by learning different attention patterns.

### Why Multiple Heads Are Better

Using multiple heads lets the Transformer capture multiple relationships **simultaneously**, rather than forcing one attention mechanism to learn everything.

- The outputs from all attention heads are later **combined** (concatenated and linearly transformed) to produce a richer, more informative representation of every word.
- This gives the model a much deeper understanding of the sentence than a single attention head could provide.

> 📌 The original Transformer ("Attention Is All You Need," 2017) used **8** attention heads, though modern Transformer models often use many more (12, 16, 32, or more, depending on model size).

### Complete Flow for Multi-Head Attention

Walking through it for the word "do," with (say) 8 attention heads:

1. The one-hot vector for "do" → embedding layer → embedding vector.
2. Embedding vector + positional encoding vector → input representation.
3. Instead of **one** set of `(W_Q, W_K, W_V)`, Multi-Head Attention uses **multiple independent sets** — one set per head. With 8 heads, that's 8 different Query matrices, 8 different Key matrices, and 8 different Value matrices.
4. The same input representation of "do" is multiplied separately by each head's weight matrices, producing a **different** Query, Key, and Value vector for every head.
5. **Within each head**: the Query vector of "do" is compared (dot product) against the Key vectors of all words in the sequence → Softmax → normalized attention weights → each weight multiplied by its corresponding Value vector → summed to produce a context vector **for that head**.
6. Since every head has its own learned weight matrices, each head produces its **own** context vector for the same word.
7. These context vectors from all heads are **concatenated** into one larger vector.
8. This concatenated vector is multiplied by another learned weight matrix `W_O` (the **output projection matrix**), producing the final context-aware representation of the word.
9. This final vector — combining information learned by all attention heads — is passed to the next layer of the Transformer encoder.

---

## Residual Connections & Layer Normalization for Self-Attention

*Reference: [ChatGPT conversation on residual connections and normalization here](https://chatgpt.com/s/t_6a5f90aae3d481918e2ef5715880ad85)*

<img width="1305" height="758" alt="image" src="https://github.com/user-attachments/assets/f1b06763-e57e-4687-97c4-015f2dad9528" />


> 💡 **Reading the diagram's playful annotations**: "We are big, fat numbers now. About to overflow. Normalize us!" — after the self-attention weighted sums accumulate, values can grow large, so **normalization keeps them well-scaled**. "Hey! Remind me what my position was!" — the **residual connection** skips the original positional-encoded input straight past the self-attention block, adding it back in so positional information isn't lost. "Also, deal with the vanishing gradients" — as with deep FFNNs and stacked RNNs, residual connections here serve the same purpose: letting gradients flow more easily through many layers.

This mirrors the residual connection concept covered earlier for RNNs — `y = F(x) + x` — applied here to the self-attention sub-layer output, followed by a **Normalize** step.

---

## Summary: What Does the Encoder Do?

The encoder transforms each input word into a **rich, context-aware representation** that captures what the word is, where it appears, and how it relates to every other word in the sentence.

1. First, it converts words into dense numerical vectors using **word embeddings**.
2. Then adds **positional encoding**, so the model knows the order of the words.
3. Next, **self-attention** allows each word to interact with all other words — learning contextual relationships and producing a representation reflecting the word's meaning *within* the sentence, rather than in isolation.
4. Finally, **residual connections** preserve the original input information and improve gradient flow, while **layer normalization** stabilizes training by keeping vector values well-scaled.

Because self-attention processes all words simultaneously, the encoder can compute these contextual representations **in parallel** — making Transformers both efficient and powerful.

---

## The Decoder

The decoder has its own **separate** Query (`W_Q`), Key (`W_K`), and Value (`W_V`) matrices for self-attention — different from those used by the encoder.

Unlike the encoder, the decoder generates the output sentence **one word at a time** (auto-regressive generation). Therefore, at any time step `t`, it can perform self-attention only over the **previously generated words and the current word**, since future words have not yet been generated.

### Masked Self-Attention

This restriction is enforced using **masked self-attention**, where future positions are hidden (masked) so the decoder cannot attend to them — even during training, when the correct future words are already known (**teacher forcing**).

### Encoder–Decoder Attention (Cross-Attention)

In addition to masked self-attention, the decoder also contains **another independent** set of Query, Key, and Value matrices for **Encoder–Decoder Attention (Cross-Attention)** — allowing it to attend to the encoder's output while generating the target sentence.

### Building the Intuition: Adding Cross-Attention

<img width="1214" height="718" alt="image" src="https://github.com/user-attachments/assets/e746dc05-f64e-47b7-9ce0-54c154c25c86" />


In this simplified diagram: the **encoder** (blue, left) processes "She goes home" — note that the encoder's own outputs produce no loss/output directly (they only exist to be attended to later). The **decoder** (green, right) generates the Urdu translation one token at a time, starting from `<SOS>`, with each decoder step reaching back to attend over **all** encoder positions (the horizontal connecting lines) — this is exactly the cross-attention mechanism.

### Full Reference Chain for the Decoder Walkthrough

*Reference 1: [Decoder f1 figure — producing normalized decoder output](https://chatgpt.com/s/t_6a5f97f727048191ae7eee0c9bff8e9d)*

*Reference 2: [Continuation — Encoder–Decoder Attention (Cross-Attention)](https://chatgpt.com/s/t_6a5fa3fb8304819183ca8c74ad0d00c5)*

*Reference 3: [Continuation — predicting the second word](https://chatgpt.com/s/t_6a5fa657073881918382f61c8e8ae668)*

> 💡 It's worth also viewing the **official slides** at the decoder portion directly, since these diagrams build on each other step-by-step across several slides.

### Walkthrough: Ready for the First Output Word?

<img width="1179" height="759" alt="image" src="https://github.com/user-attachments/assets/bede02a2-c13a-425b-9235-ca2c6763ff73" />


This diagram shows the **full encoder stack** (3 positions: "do," "eat," "not," `<s>`) alongside the **decoder**, just as the decoder is about to attend to the encoder for the very first time — each encoder position has already computed its own self-attention (`SA`) output via Q/K/V, and the decoder has computed its own masked self-attention over the target-side tokens generated so far.

### Walkthrough: Adding in Encoder-Decoder Attention

<img width="1173" height="652" alt="image" src="https://github.com/user-attachments/assets/5e73b7a5-d0e5-42ef-8405-8c43f5f27fe7" />


Here, the red arrows/circles highlight exactly which encoder Key and Value vectors (`K`, `V` — circled in the encoder blocks) the decoder's masked self-attention output attends to, forming the **cross-attention** connection. The decoder's Query (from its own masked self-attention output) reaches across to the encoder's Keys and Values — this is the literal mechanism of "the decoder looking at the encoder."

### Walkthrough: Predicting the Second Word




This shows the same architecture one step further along in generation — having already produced the first output word, the decoder now includes it as part of its own masked self-attention context (alongside `<s>`) while continuing to cross-attend to the full encoder output, to predict the *second* target word.

### Walkthrough: The Full Decoder Summary




This final diagram shows the **complete decoder pipeline** end-to-end: encoder blocks (bottom-left, each computing Key/Value pairs) feed into the decoder's **Encoder-Decoder Attention** (`EDA`) blocks, which combine with the decoder's own masked self-attention outputs, get normalized, pass through an output projection, and finally through **Softmax** to produce the probability distribution over the target vocabulary — here predicting the Urdu tokens for the translation of "do eat not."

---

## Summary: What Does the Decoder Do?

The decoder generates the target sentence one word at a time, using **auto-regressive generation**.

1. It first converts the previously generated target words (or the `<START>` token initially) into word embeddings, adds positional encodings, and applies **masked self-attention** — where each word can attend only to itself and previously generated words, while future words are masked.

2. The resulting representation is passed to **Encoder–Decoder Attention (Cross-Attention)**, where the **Query comes from the decoder**, and the **Keys and Values come from the encoder outputs** — letting the decoder focus on the most relevant parts of the input sentence.

3. After **Residual Connections** and **Layer Normalization**, the final decoder representation is passed through a **Linear layer** and **Softmax** to predict the next target word.

4. This process repeats until the model generates the **end-of-sequence** (`<EOS>`) token, producing the complete output sentence.
