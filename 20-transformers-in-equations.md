# Transformers in Equations: The Mathematics of Transformers

*This is the equations-first companion to the earlier intuition-based Transformer notes — same architecture, now traced through with exact formulas, matrix shapes, and dimensions.*

### References Used in This Lecture

- [Columbia University — Transformers notes](http://www.columbia.edu/~jsl2239/transformers.html)
- [The Math Behind Transformers — Medium](https://medium.com/@cristianleo120/the-math-behind-transformers-6d7710682a1f)
- [Lilian Weng — The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Understanding Transformers: A Step-by-Step Math Example — Medium](https://medium.com/@fareedkhandev/understanding-transformers-a-step-by-step-math-example-part-1-a7809015150a)
- [e2eml.school — Transformers](https://e2eml.school/transformers.html)
- [Lena Voita — Sequence to Sequence and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)

---

## The Full Architecture, At a Glance

<img width="1650" height="1275" alt="image" src="https://github.com/user-attachments/assets/d5041426-f7c6-4245-bb5e-86a63c946c63" />


*(Source: [Lilian Weng — The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/))*


---

## Step 1: Input Embedding + Positional Encoding

Given input token IDs `(t_1, t_2, ..., t_n)`, look up each token's embedding and add positional encoding:

```
X = TokenEmbedding(t) + PE
```

Where `PE` uses **fixed sinusoidal functions** (no learned parameters):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

This gives the model a sense of word order **without any learned parameters**. The result `X` is a matrix of shape `n × d`, where `d = 512` in the base model.

### Token Embedding, Formally

For an input token sequence `t_1, t_2, ..., t_n`, each token `t_i` from vocabulary `V` is mapped to a continuous embedding vector `x_i` of length `d`, using a layer with weight matrix `W_E`:

```
x_i = t_i · W_E
```

Where:
```
t_i ∈ {0,1}^(1×|V|)     (one-hot vector)
W_E ∈ ℝ^(|V|×d)         (embedding matrix)
x_i ∈ ℝ^(1×d)           (resulting embedding)
```

### Positional Encoding, Formally

Since the Transformer has no inherent recurrence, we add a positional encoding `PE(pos, i)` to inject order information. A common choice:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Combined**: the final input for position `pos` is:
```
X_pos = x_pos + PE(pos) ∈ ℝ^(1×d)
```

> 💡 **Intuition**: embedding translates discrete tokens into vectors that capture semantic meaning, while positional encodings supply information about token order, letting the model know which token comes first, second, and so on.

---

## Deep Dive: Positional Embeddings

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))   if i is even
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))   if i is odd
```

Where:
- `pos` is just the integer position in the sequence (e.g., 0, 1, 2, ...).
  > Example: *"The mad hatter wore a crazy hat"* → `pos: 0 1 2 3 4 5 6`
- Each `i` in `PE(pos, i)` corresponds to an embedding dimension, and samples a different sinusoid — `i ∈ [0, d]`.
- `d_model` is the dimensionality of the model's embeddings (e.g., 512, 300, etc.).
- The exponent `2i/d_model` tells you how "fast" or "slow" the wave oscillates.

### Intuition: Radio Stations

Think of each dimension as a separate **"radio station."**

- For each dimension `i`, the network "dials in" to a particular frequency of a sine/cosine wave.
- **Low-index** dimensions correspond to **high-frequency** waves; **higher-index** dimensions correspond to **lower-frequency** waves.

**Every position samples these waves at a specific point in time**:
- If we fix `i` (pick one dimension) and vary the token position `pos`, we trace out a sinusoid over positions.
- Sine and cosine are just phase-shifted versions of each other, so splitting them into even (sine) and odd (cosine) channels covers both "phases" of the same underlying frequency.

**Combining many frequencies yields a unique pattern at each position**:
- By stacking all these sine/cosine values together into a vector, each position `pos` corresponds to a unique pattern across the dimensions.
- The Transformer can then learn to interpret these patterns, letting it know "where" in the sequence each token occurs — both **absolutely** and **relative to other tokens**.

### The Frequency Term, Formally

If we define:
```
ω_i = 1 / 10000^(2i/d_model)
```

then each dimension `i` is really sampling one of the two sinusoids: `sin(pos·ω_i)` or `cos(pos·ω_i)`.

- **Small `i`** (near 0): the exponent `2i/d_model` is small, so `10000^(2i/d_model) ≈ 1`. The frequency `ω_i` is relatively large → **fast oscillations** (short-period sinusoids).
- **Large `i`**: the exponent is larger, so `10000^(2i/d_model) ≫ 1`. The frequency `ω_i` is smaller → **slower oscillations** (long-period sinusoids).

Hence, across the different dimensions, we get a range of frequencies spanning from "fast" (position changes cause frequent ups and downs) to "slow" (position changes cause a gentle drift). **Any specific position `pos` ends up having a unique fingerprint across all these frequencies.**

### Why 10,000, Specifically?

The choice of 10,000 is largely a design/hyperparameter choice from the original "Attention Is All You Need" paper. It ensures that, as `i` goes from 0 up to `d_model`, we span a wide but practical range of frequencies for most sequence lengths of interest.

**Intuitive reasons for choosing 10,000 (rather than 1,000 or 1,000,000):**

- **Wide Frequency Coverage**: each dimension `i` should correspond to a different timescale/"wave speed."
  - Too small a base → we won't capture long-range dependencies (low-frequency waves would still be too fast).
  - Too large a base → the lowest-frequency wave might be so slow that many practical sequence lengths never complete even a single oscillation.
- **Easy to Work With / Reasonable Default**: the authors needed a number "big enough" to cover typical sequence lengths (a few hundred to a couple thousand tokens).
- Using a round number like 10,000 is convenient and intuitive: small `i` → fast sinusoids (short-range structure); larger `i` → slower sinusoids (longer-range structure).
- **It's a Simple Hyperparameter**: you could pick a different base or tune it — Transformer performance doesn't drastically hinge on this exact value of 10,000. In practice, it just works well and is a widely used default.

### Visualizing the Positional Encoding Grid

<img width="1650" height="1275" alt="image" src="https://github.com/user-attachments/assets/86ffa74b-bb91-477b-ba7a-958f37d5ea84" />


> 💡 **Reading this diagram**: each column of waves (labeled `i=0`, `i=1`, `i=2`, `i=3`... up through `i=10, 20, 30`) shows a sine/cosine wave at a different frequency, plotted against word position. The table below shows how each cell's value is `x_{pos,i} + PE(pos,i)` — the word embedding value at that dimension, **plus** the positional encoding value for that position and dimension. This is the literal, cell-by-cell picture of "embedding + positional encoding," applied to the example sentence "The mad hatter wore a crazy..." across positions 0 through 5.

---

## Step 2: Encoder Block (repeated N = 6 times)

Each encoder block has **two sub-layers**.

### Sub-layer 1: Multi-Head Self-Attention

For each head `i` (`i = 0` to `7`):
```
Q_i = X · W_i^Q   (shape: n × d_k)
K_i = X · W_i^K   (shape: n × d_k)
V_i = X · W_i^V   (shape: n × d_v)

head_i = softmax( (Q_i · K_i^T) / √d_k ) · V_i
```

- The softmax is applied **row-wise**, so each token gets a probability distribution over all other tokens.
- Division by `√d_k` (= `√64 = 8` in the base model) prevents the dot products from getting too large, which would push softmax into regions with vanishing gradients.

**Then concatenate and project:**
```
MultiHead(X) = Concat(head_0, head_1, ..., head_7) · W^O
```

**Wrapped with residual connection and LayerNorm:**
```
X′ = LayerNorm(X + MultiHead(X))
```

### Queries, Keys, and Values — The Concept

In Self-Attention, we extract three types of vectors/encodings from each token in the sequence — each with a different role:

1. **Query** — *"Here's what I'm interested in…"*
2. **Key** — *"This is what I have…"*
3. **Value** — *"If you find me interesting, here's what I will communicate to you…"*

> 💡 **Search engine analogy**: When you search for videos on YouTube, the search engine maps your **query** (text in the search bar) against a set of **keys** (video title, description, etc.) associated with candidate videos in their database, then presents you with the best-matched videos (**values**).

*(Source: [Stats StackExchange — What exactly are Keys, Queries, and Values in Attention?](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms))*

### Self-Attention: Q, K, V — Formally

Starting from the positionally encoded embeddings `X_1, X_2, ..., X_n`, where `X_pos ∈ ℝ^(1×d)`.

The three vectors are extracted by (learnable) projections:
```
k_pos = X_pos · W^K,   W^K ∈ ℝ^(d×d_k)
q_pos = X_pos · W^Q,   W^Q ∈ ℝ^(d×d_k)
v_pos = X_pos · W^V,   W^V ∈ ℝ^(d×d_v)
```

- It's helpful to keep `d_q = d_k`.
- For simplicity of derivations, we'll keep `d_q = d_k = d_v = d`.

### Self-Attention: Dot-Product

Similar to the search engine analogy: when looking for a good match, we focus on the query and the key.

- It only makes sense to compare an individual query vector against **all** key vectors (imagine if Google only searched through one webpage for your query!).
- Since we're dealing with vectors, we compute a compatibility/affinity score with **dot products**:
```
q_1·k_1, q_1·k_2, q_1·k_3, ..., q_1·k_n
q_2·k_1, q_2·k_2, q_2·k_3, ..., q_2·k_n
...
q_n·k_1, q_n·k_2, q_n·k_3, ..., q_n·k_n
```

### Self-Attention: Vectorizing the Approach

Before diving deeper, it's cleaner to shift from individual tokens to dealing with entire sequences at once — vectorizing makes the equations much simpler.

Instead of dealing with individual input vectors, concatenate them all into one large matrix; similarly, get Query, Key, and Value matrices in one clean expression:

```
X = [X_1; X_2; ...; X_n] ∈ ℝ^(n×d)

K = X·W^K ∈ ℝ^(n×d_k),   W^K ∈ ℝ^(d×d_k)
Q = X·W^Q ∈ ℝ^(n×d_k),   W^Q ∈ ℝ^(d×d_k)
V = X·W^V ∈ ℝ^(n×d_v),   W^V ∈ ℝ^(d×d_v)
```

Now, we can get affinity scores for **all pairs at once**, via one matrix multiplication:
```
QK^T ∈ ℝ^(n×n)
```

### Self-Attention: Scores to Weights

Since we have affinity scores for each key relative to a given query, they should be normalized to be interpreted as weights (for a weighted average later).

- Recall: for classification, we pass the logits of a neural network through Softmax to get probabilities.
- **Extra step**: divide the scores by `√d_k` (the dimensionality of the key vector) — this helps with stable gradients, otherwise the Softmax input would be very large.

```
softmax( QK^T / √d_k ) ∈ ℝ^(n×n)
```

Where for each row `pos`, the resulting weights `s_{pos,·}` sum to 1.

### Self-Attention: Multiply With Values

Now that scores are normalized to a sensible range, we can aggregate information from other tokens into our query token, based on the scores found — this is just a matrix multiplication with the Value matrix:

```
H = softmax(QK^T / √d_k) · V ∈ ℝ^(n×n) × ℝ^(n×d_v) = ℝ^(n×d_v)
```

We get **one `d_v`-sized "smoothie" for every token** `X_pos`.

> 💡 **Key observation**: the Attention mechanism results in a series of weighted averages of the rows of `V`, where the weighting depends on the input queries and keys. Each of the `n` queries in `Q` results in a specific weighted sum of the value vectors. **Of note**: there are no learnable parameters in this particular procedure (the softmax/weighting step itself) — it's entirely comprised of matrix and vector operations. (The learnable parameters are in the `W^Q`, `W^K`, `W^V` projection matrices used *before* this step.)

---

## Multi-Head Attention

Multi-Head Attention is an extension of Scaled Dot-Product Attention, in which **linear transformations are run on the queries, keys, and values** to yield multiple sets of inputs to the attention function.

- The attention function is computed **in parallel**, given each of these sets of inputs.
- The results are **concatenated** side-by-side into one matrix.
- The final result is obtained via a linear transformation of the concatenated matrix, to get a matrix with the desired dimensionality.

```
Multihead(Q, K, V) = Concat(head_1, head_2, ..., head_h) · W^O
                    ∈ ℝ^(n×h·d_v) × ℝ^(h·d_v×d) = ℝ^(n×d)
```

Each head `i` is the result of running Scaled Dot-Product Attention on the `i`-th set of transformed queries, keys, and values:
```
head_i = Attention(X·W_i^Q, X·W_i^K, X·W_i^V)
```

Where `X ∈ ℝ^(n×d)`.

**Given a hyperparameter `h` (number of attention heads):**
```
W_i^Q ∈ ℝ^(d×d_k),   W_i^K ∈ ℝ^(d×d_k),   W_i^V ∈ ℝ^(d×d_v),   W^O ∈ ℝ^(h·d_v×d),   head_i ∈ ℝ^(n×d_v)
```

`head_i` contains `d_v`-sized vectors for all `n` tokens.

### Visualizing Multiple Heads Focusing on Different Things

<img width="420" height="223" alt="image" src="https://github.com/user-attachments/assets/0421ec05-73fa-4f22-8fcb-3c5b4cbfb9e3" />


*(Source: [Lena Voita — seq2seq and attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html))*

### Sanity Check: Do the Dimensions Actually Work Out?

- Each matrix `head_i` will have the same number of **rows** as `Q·W_i^Q`, and the same number of **columns** as `V·W_i^V`.
- Since `Q·W_i^Q ∈ ℝ^(n×d_k)` and `V·W_i^V ∈ ℝ^(n×d_v)`, this means `head_i ∈ ℝ^(n×d_v)`.
- When we concatenate `h` of these `head_i` together, we get a matrix in `ℝ^(n×h·d_v)`.
- Multiplying with `W^O` (∈ `ℝ^(h·d_v×d)`) yields a matrix in `ℝ^(n×d)`.
- This makes sense — we started with `n` queries in `Q`, and end up with `n` responses in the output of the Multi-Head operator.

> 💡 Notice: each head computation has a **different** linear transformation for the key, query, and value matrices. Each of these transformations is learned during training — this is exactly what lets different heads specialize on different relationship types.

### Multi-Head Attention — Applications

Multi-head attention is used in **three** ways in the Transformer:

1. **Computing self-attention in the encoder blocks** — the encoder's bi-directional self-attention (already covered above).

2. **Computing self-attention with masking in the decoder blocks** — the decoder uses masked self-attention due to autoregressive generation, i.e., it only has access to previous states.

3. **Encoder-decoder attention within the decoder blocks** — this uses `Q` from the decoder, while `K` and `V` come from the encoder.

---

## Residual Connections

To stabilize and speed up training, each sub-layer in the encoder (multi-head attention and feed-forward network) is followed by a **residual connection** and **layer normalization**.

- Residual connections help mitigate the vanishing gradient problem and allow for training deeper networks.
- **The basic idea**: add the input of the sub-layer to its output:

```
Output = LayerNorm(x + subLayer(x))
```

Where:
- `x` is the input to the sub-layer (could be the multi-head attention mechanism or the feed-forward network).
- `Sublayer(x)` is the output from the sub-layer after processing `x`.
- `x + Sublayer(x)` is the residual connection, which adds the original input `x` to the sub-layer's output — helping gradients flow more easily through the network, preventing them from becoming too small (vanishing gradient) or too large (exploding gradient).

---

## A Few Extras

- The original paper used a vocabulary of **37,000 tokens**, and dealt with longer inputs and outputs.
- To make this work, they had to **normalize often** — the original paper normalizes after positional encodings **and** after self-attention.
- To give the Transformer more weights and biases to deal with complicated data, they added additional neural networks with hidden layers to both the encoders and decoders (i.e., the Position-wise Feed-Forward Networks, covered next).

---

## Sub-layer 2: Position-wise Feed-Forward Network

```
FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2
```

- This is just **two linear layers with ReLU in between**, applied identically to each token position.
- `W_1` projects from 512 up to 2048 (**expanding**), and `W_2` projects back down from 2048 to 512.
- This expansion gives the model a larger space to do non-linear computation before compressing back.
- Again wrapped with residual + LayerNorm:
```
X″ = LayerNorm(X′ + FFN(X′))
```

This `X″` becomes the input to the next encoder block. After all 6 blocks, the final output is the **encoder representation** — call it **`Enc`**.

> 💡 **A useful one-line framing**: Attention = *"look at other tokens and gather information."* FFN = *"take a moment to think and process this information."* The two sub-layers do genuinely different jobs — attention is about **communication between tokens**, while the FFN is about **individual, position-wise computation**.

**FFN, restated with explicit ReLU:**
```
FFN(x) = ReLU(x·W_1 + b_1) · W_2 + b_2
       = max(x·W_1 + b_1, 0) · W_2 + b_2
```

Here, `x` is the input from the attention mechanism, `W_1` and `W_2` are weight matrices, and `b_1` and `b_2` are biases.

> 💡 **ReLU as a gatekeeper**: ReLU passes positive values unchanged, but blocks negatives, turning them to zero. This introduces non-linearity, allowing the network to learn more complex patterns than a purely linear transformation ever could.

---

## Layer Normalization

The "Norm" part in "Add & Norm" denotes **Layer Normalization**.

- It independently normalizes the vector representation of **each example** in a batch — controlling "flow" to the next layer.
- Layer normalization improves convergence stability, and sometimes even quality.
- We normalize the vector representation of **each token**.
- `LayerNorm` also has **trainable parameters** — scale (`γ`) and bias (`β`) — used after normalization to rescale the layer's outputs.
- Note: `μ` (mean) and `σ` (standard deviation) are evaluated **per example**, but `γ` and `β` are the **same across examples** — these are layer parameters, shared and learned.

### The Formula

```
LayerNorm(z) = ((z − μ) / (σ + ε)) · γ + β
```

- `z` is the input to layer normalization — in this context, `z = x + Sublayer(x)`.
- `μ` is the mean of the input `z`:
```
μ = (1/N) · Σ z
```
where `N` is the number of elements in `z` (the dimension of the hidden layer).

- `σ` is the standard deviation of the input `z`:
```
σ = √( (1/N) · Σ (z − μ)² )
```

- `ε` is a small constant added for numerical stability, to prevent division by zero.
- `γ` is a **learned scaling parameter** that allows the model to adjust the normalized values.
- `β` is a **learned shifting parameter** that allows the model to adjust the normalized values.

The normalized output ensures each sub-layer receives input with a **mean of 0 and variance of 1**, helping maintain stable gradients and improving the efficiency and effectiveness of training.

---

## The Decoder (Training)

## Step 3: Output Embedding + Positional Encoding

During training, the target sequence `(y_1, y_2, ..., y_m)` is embedded the same way:
```
Y = TokenEmbedding(y) + PE
```

The decoder uses the same embedding method and the same sinusoidal PE formula (though typically a **separate** learned embedding matrix for the target vocabulary, if the source and target languages differ).

### Target Representations, Formally

For the target token sequence `<s>, x'_1, x'_2, ..., x'_m`, each token `x'_i` from vocabulary `V'` is mapped to a continuous embedding vector `X'_i` of length `d`, using weight matrix `W'_E`:

```
X'_i = x'_i · W'_E

x'_i ∈ {0,1}^(1×|V'|)
W'_E ∈ ℝ^(|V'|×d)
X'_i ∈ ℝ^(1×d)
```

**Positional encoding**, identical formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Combined**: `Y_pos = X'_pos + PE(pos) ∈ ℝ^(1×d)`

---

## Step 4: Decoder Block (repeated N = 6 times)

Each decoder block has **three sub-layers**.

### Sub-layer 1: Masked Multi-Head Self-Attention

Same math as encoder self-attention, but with a crucial **mask**: before the softmax, positions are prevented from attending to future tokens, by setting those entries to `-∞`:

```
head_i = softmax( (Q_i·K_i^T)/√d + Mask ) · V_i
```

where `Mask` is an **upper-triangular matrix of `-∞` values**. This ensures that when predicting token `t`, the decoder can only see tokens `1` through `t-1`. Without this, the model would **"cheat"** during training by looking ahead.

```
Y′ = LayerNorm(Y + MaskedMultiHead(Y))
```

### Masked Self-Attention — Why It's Needed

In the decoder, self-attention differs from the encoder's:

- The **encoder** receives all tokens at once, and tokens can look at **all** tokens in the input sentence.
- In the **decoder**, we generate one token at a time — during generation, we don't yet know which tokens we'll generate in the future.

To forbid the decoder from looking ahead, it uses **masked self-attention**: future tokens are masked out.

### Masked Self-Attention, Formally

**Projections:**
```
Q = Y·W^Q,   K = Y·W^K,   V = Y·W^V
```

**Masked Attention**: a mask `M` (which assigns a large negative value to positions corresponding to "future" tokens) is added to the dot-products:
```
MaskedAttention(Q, K, V) = softmax( (QK^T)/√d_k + M ) · V
```

**Residual connection and norm:**
```
Y′ = LayerNorm(Y + MaskedAttention(Q, K, V))
```

> 💡 **Intuition**: masking ensures the decoder can only attend to previous tokens (and the current token) when predicting the next token — preserving the autoregressive property required during generation.

### The Mask Matrix, Worked Example

Let `Q = Y·W^Q`, `K = Y·W^K`, `V = Y·W^V`. Normally, given `head` defined as above:
```
Attention(Q, K, V) = softmax(QK^T/√d) · V
```

Since softmax is defined as `f(x_i) = e^{x_i} / Σ_j e^{x_j}`, one way to mask the inputs is to simply add a matrix `M` — consisting of `0`s in its lower triangle and `-∞`s everywhere else — to the raw score matrix:

```
        [q1·k1  q1·k2  ...  q1·kn]     [ 0   -∞  ...  -∞]     [q1·k1  -∞    ...  -∞  ]
QK^T =  [q2·k1  q2·k2  ...  q2·kn]  +   [ 0    0  ...  -∞]  =  [q2·k1 q2·k2 ...  -∞  ]
        [  ⋮      ⋮    ⋱     ⋮  ]     [ ⋮    ⋮  ⋱    ⋮ ]     [  ⋮      ⋮   ⋱    ⋮   ]
        [qn·k1  qn·k2  ...  qn·kn]     [ 0    0  ...   0]     [qn·k1  qn·k2 ...  qn·kn]
```

Running softmax on each row has the effect of converting all the `-∞` cells to `0`, leaving only the valid attention terms.

### Why This Works: The Softmax Math

Starting from the softmax definition:
```
f(x_i) = e^{x_i} / Σ_j e^{x_j}
```

When `x_i = -∞`:
- **Numerator**: `e^{-∞} = 1/e^∞ = 1/∞ = 0`.
- **Denominator**: the sum still has finite terms from the non-masked positions (those with actual dot-product values), so it's some positive finite number.
- **Result**: `0 / (finite number) = 0`.

The masked position contributes **zero** attention weight, and the remaining (non-masked) positions get renormalized to sum to 1 among themselves. So the softmax naturally redistributes all the probability mass onto only the valid positions — exactly what's needed for **causal masking**.

---

### Sub-layer 2: Cross-Attention (Encoder-Decoder Attention)

This is the **bridge** between encoder and decoder. The queries come from the decoder, but the keys and values come from the encoder output:

```
Q_i = Y′ · W_i^Q      (from decoder)
K_i = Enc · W_i^K     (from encoder)
V_i = Enc · W_i^V     (from encoder)

head_i = softmax( (Q_i·K_i^T)/√d ) · V_i
```

This lets **every decoder position attend to every encoder position** — this is how the decoder "reads" the input. For example, when generating the Urdu word for "cat," cross-attention can focus heavily on the English word "cat" in the encoder output.

```
Y″ = LayerNorm(Y + CrossAttention(Y′, Enc))
```

### Cross-Attention, Formally

**Query from Decoder, Keys/Values from Encoder:**
```
Q     = Y · W^Q
K_enc = EncoderOutput · W^K
V_enc = EncoderOutput · W^V
```

**Attention calculation:**
```
Attention(Q, K_enc, V_enc) = softmax( (Q·K_enc^T)/√d_k ) · V_enc
```

**Residual and Norm:**
```
Y″ = LayerNorm(Y′ + Attention(Q, K_enc, V_enc))
```

> 💡 **Intuition**: this step lets the decoder "look back" at the encoder's output (which contains information about the entire source sequence), and align the generated output with the input.

---

### Sub-layer 3: Feed-Forward Network

Identical structure to the encoder FFN:
```
Y′′′ = LayerNorm(Y″ + FFN(Y″))
```

This `Y′′′` becomes the input to the next decoder block.

**FFN application:**
```
FFN(y) = max(0, y·W_1 + b_1) · W_2 + b_2
```

**Residual and Norm:**
```
Y′′′ = LayerNorm(Y″ + FFN(Y″))
```

> 💡 **Intuition**: just like in the encoder, the FFN adds additional non-linearity and lets each position's representation be further transformed independently.

---

## Step 5: Final Output Projection

After all 6 decoder blocks, the output is projected to vocabulary size and converted to probabilities:

```
logits = Y′′′ · W_vocab     (shape: m × vocab)
P = softmax(logits)         (per position)
```

The model is trained to maximize the probability of the correct next token at each position, using **cross-entropy loss**. The weight matrix `W_vocab` is typically **tied** with the token embedding matrix (transposed) — the same representation is used for input and output (see "Weight Tying" from the earlier notes).

### Output Projection and Softmax, Formally

**Projection to vocabulary space:**
```
Logits = Y′′′ · W_P + b_P
```

**Probability distribution** — applying softmax gives the probabilities for each token:
```
Prob(w) = softmax(Logits)
```

> 💡 **Intuition**: the projection transforms the hidden representations into scores for each word in the vocabulary, and softmax converts these scores into probabilities, which are then used to choose (or sample) the next token in the sequence.

---

## How the 6 Encoders and 6 Decoders Stack

**Encoder stacking is simple and serial:**
- Encoder block 1's output feeds into encoder block 2's input, block 2 feeds into block 3, and so on.
- Only the output of the **last** encoder (block 6) matters — intermediate outputs are discarded.

**Decoder stacking is also serial for the self-attention and FFN**, but:
- The **cross-attention** in *every* decoder block receives the **same** encoder output — i.e., the final output of encoder block 6.
- So it's **not** the case that decoder block 1 connects to encoder block 1, decoder block 2 to encoder block 2, etc.
- Instead, **all 6 decoder blocks read from the same encoder representation.**

> 💡 This design means the encoder can process the entire input in parallel (no masking needed), while the decoder generates tokens autoregressively — one at a time during inference — using the previously generated tokens as input.

---

## Encoder and Decoder Blocks — The Full Flow Summary

### Encoder Block (2 sub-layers)

```
X → MultiHeadAttention → + X → LayerNorm → FFN → + → LayerNorm → output
    ↑_________________________|                    ↑________|
    (residual)                   (residual)
```

That's it. **No cross-attention, no masking** — those are decoder-only features.

### Decoder Block (3 sub-layers)

A decoder block, by contrast, has three sub-layers:
1. Masked multi-headed self-attention
2. Multi-headed cross-attention
3. Position-wise FFN

So the decoder is the more complex of the two.

> 💡 **The encoder is surprisingly simple.** It's just self-attention and a feed-forward network, repeated 6 times — and it's that repetition through depth that gives it the power to build rich contextual representations.

---

## Closing the Loop

![The Transformer — final view](./images/transformer-full-architecture-overview.jpg)

*(Source: [Lena Voita — seq2seq and attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html))*
