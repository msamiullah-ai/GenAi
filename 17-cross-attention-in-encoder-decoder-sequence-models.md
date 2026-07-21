# Cross-Attention in Encoder-Decoder Sequence Models

---

## The Information Bottleneck Problem

In standard sequence-to-sequence models, the **information bottleneck** occurs because the entire input sequence must be compressed into a single, fixed-length **context vector** (often the final hidden state of the encoder). This forces the model to cram the meaning of a lengthy, variable-length sequence into a small, fixed-size representation.

- Because of this fixed capacity constraint, the model struggles to remember details from the **earlier** parts of long input sequences.
- As the decoder generates output, it must rely exclusively on this single, finite vector, rather than having continuous, dynamic access to the original input tokens.

This bottleneck limits performance and causes degradation on long-context tasks (like translating lengthy paragraphs). It is precisely what the **attention mechanism** was designed to solve — by letting the decoder dynamically weigh and access all encoder states directly, instead of being confined to a single vector.

### Three Ways to Frame the Problem

- **Information Bottleneck**: the model attempts to compress the entire source input (which can be a long passage of text) into a single, fixed-size context vector.

- **Context Dilution**: due to multiple non-linear transformations and multiplications as data passes through the sequence model (RNN or LSTM), the initial context becomes progressively **"diluted"** or lost.

> 💡 **Memory Limitations analogy**: this is like asking someone to read a long passage, close the book, and then translate it purely from memory — inefficient, and it leads to poor performance on longer, complex sequences.

---

## Attempt 1: Feed the Context Vector to Every Decoder Step

*Reference: [ChatGPT conversation on Attempt 1](https://chatgpt.com/s/t_6a5e3d961738819188a74bf33bc9c742)*

**The simplest fix for the bottleneck**: instead of using the context vector `c` only to *initialize* the decoder, also feed it as an additional input at **every** decoder timestep.

<img width="1347" height="774" alt="image" src="https://github.com/user-attachments/assets/adbd4532-7e2a-44a4-966a-42f7b4a20fac" />


**Architecture change**: at each decoder step `t`:
```
h_t^dec = RNN([emb(y_{t-1}) ; c], h_{t-1}^dec)
```
where `c = h_n^enc` (the final encoder hidden state) and `[ ; ]` denotes concatenation.

The decoder now sees the source summary at **every** step, not just at initialization.

### Strengths

- **No memory fade in the decoder**: the decoder doesn't have to carry the original input sentence entirely in its own recurrent memory chain, because `c` serves as a **constant, repeated reminder** at every step.
- **Combined information**: each decoder step merges its evolving output history with the original source context.
- **Better performance**: empirically, this performs significantly better than designs that only use `c` to initialize step 0.

### Weaknesses

- **Static context** (`c` is identical for every word): the vector `c` remains fixed. Whether the decoder is generating a subject, a verb, or an object, it sees the **exact same** vector `c` — it cannot dynamically focus on different parts of the input sentence.
- **Encoder compression problem**: the encoder is still forced to crush an entire sentence into a single vector of fixed dimension.
- **Capacity bottleneck**: for long sentences, the specific words most relevant to the current output word get buried or diluted inside the averaged vector.

---

## The Bridge to Attention: From Static to Dynamic Context

**The question that motivates attention**: what if we replace the *same* summary being fed into each decoder step, with a **customized summary** according to the word currently being generated (or what's been generated so far)?

> 💡 **Example**: if a verb is currently being generated, the model should focus on the verb, raising its importance in the context vector at that specific step.

**The core idea**: give higher weights to the things that are most important to the current context and current generation — **this is the idea of Attention.**

```
Static vector  →  Dynamic vector
```

---

## Attempt 2: Pass ALL Encoder Hidden States

*Reference: [ChatGPT conversation on Attempt 2](https://chatgpt.com/s/t_6a5e43cecdb48191b4f53dba3d27a888)*

*Reference: [Selection of hidden state](https://chatgpt.com/s/t_6a5e447d097081918bbe977fc79f0501)*

Instead of compressing the source into a single vector, keep **all** encoder hidden states `s_1, s_2, ..., s_k` and make them available to the decoder at **every** step.

<img width="1370" height="776" alt="image" src="https://github.com/user-attachments/assets/4bf2d4b4-80d3-4b53-87f0-c3c403df6ecb" />


- This solves the **storage** problem: no information is lost.
- But it creates a new **selection** problem: the decoder now has `m` vectors available. At each step, **which ones matter most?**

> 💡 This selection problem is exactly what the attention mechanism answers — it's the missing piece that turns "we have all the information" into "we know which parts of it matter right now."

---

## Attention: The Mechanism

*Reference: [ChatGPT conversation on the attention mechanism](https://chatgpt.com/s/t_6a5e54fe5a90819199b2d7790834d93f)*

**Goal**: compute a **dynamic context vector** `c_t` at every decoder step.

<img width="1392" height="769" alt="image" src="https://github.com/user-attachments/assets/9a6ff0cf-3d09-4671-ad12-a4545e6ee89b" />


### Building Attention From the Bottom Up

**1. Attention Input:**
```
s_1, s_2, ..., s_m   ← all encoder states
h_t                  ← one decoder state
```

**2. Attention Scores** — *"How relevant is source token k for target step t?"*
```
score(h_t, s_k),   k = 1..m
```

**3. Attention Weights** — *"attention weight for source token k at decoder step t"* (computed via softmax):
```
a_k^(t) = exp(score(h_t, s_k)) / Σ_{i=1}^{m} exp(score(h_t, s_i)),   k = 1..m
```

**4. Attention Output** — *"source context for decoder step t"* (a weighted sum):
```
c^(t) = a_1^(t)·s_1 + a_2^(t)·s_2 + ... + a_m^(t)·s_m = Σ_{k=1}^{m} a_k^(t)·s_k
```

> 💡 **The learning intuition**: "when I am generating this word, focusing on this particular source word gives less error." When the output is good and cross-entropy loss becomes lower, the model has learned the best word(s) to focus on — there's no manual rule here, it's all learned through gradient descent, just like the rest of the network.

### Which Decoder Hidden State Is Used to Compute Attention Scores?

*Reference: [ChatGPT conversation on this exact question](https://chatgpt.com/s/t_6a5e56ce36d0819181fa57a5cf34d707)*

There are **two valid choices** here:

1. Comparing `h_t` (generated by the RNN at time `t`) with `s_k`, **OR**
2. Comparing `h_{t-1}` (generated by the RNN at time `t-1`) with `s_k`, and passing the result to the RNN's output at time `t`.

*(Reference for the diagram above: SLP3, [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/); Sequence to Sequence (seq2seq) and Attention: [lena-voita.github.io/nlp_course/seq2seq_and_attention.html](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html))*

---

## Worked Example: Encoder-Decoder With Dot Product Attention

<img width="443" height="256" alt="image" src="https://github.com/user-attachments/assets/2c485834-6379-4b61-930f-fb5bde064c35" />


**Walking through the diagram:**

1. **Encoder** processes the input sequence ("do," "eat," "not," `<s>`) one-hot token by one-hot token, producing hidden states `h_e0`, `h_e1`, `h_e2`.

2. For each encoder hidden state, compute:
   - **Dot Product or Cosine Similarity** between it and the current decoder hidden state (e.g., `h_e0 · h_d0`).
   - **Scaled similarity** via Softmax, turning raw scores into normalized weights (e.g., 0.7, 0.2, 0.1).
   - **Scaled Contribution**: each encoder hidden state scaled by its corresponding attention weight.
   - **Sum**: all scaled contributions are added together to form the dynamic context vector for that decoder step.

3. This context vector is combined with the decoder's own hidden state and passed through **Softmax** to predict the next output token — repeating at each decoder step (`h_d0` → `h_d1` → `h_d2`), with a **freshly computed** context vector at every single step.

---

## Combining the Context Vector With the Decoder Hidden State

The dynamic context vector `c^(t)` is **not** directly multiplied with the decoder hidden state `h_t`. Instead:

1. They are **concatenated** into a single vector — combining the decoder's knowledge of previously generated target words (`h_t`) with the source information selected by attention (`c^(t)`).

2. This combined vector is passed through a **learnable linear layer** (often followed by a `tanh` activation), producing an **attentional hidden state**.

3. This representation is multiplied by an **output weight matrix** and passed through **Softmax**, generating the probability distribution over the target vocabulary — predicting the next word.

**The correct process, in order:**
```
concatenation → linear projection (optional tanh) → output layer → Softmax → Final Output
```

---

## Comparing Score Functions

The purpose of a **score function** is to measure how well the current decoder hidden state matches each encoder hidden state:

```
e_tk = score(h_t, s_k)
```
- `e_tk` — raw attention score between the decoder at step `t` and the `k`-th encoder hidden state.
- `h_t` — decoder hidden state at time step `t`.
- `s_k` — hidden state of the `k`-th encoder word.

*Reference: [ChatGPT conversation comparing score functions](https://chatgpt.com/s/t_6a5e5b6d43dc81918ea4384062ba6d57)*

### A Big Problem With Raw Dot-Product Attention

*Reference: [ChatGPT conversation on the dot-product problem](https://chatgpt.com/s/t_6a5e5dd5b1e48191ae8d3bbcabefe6b7)*

### Luong Attention

*Reference: [ChatGPT conversation on Luong Attention](https://chatgpt.com/s/t_6a5e617fbdf88191ac4f5f02e4cdeeb5)*

### Bahdanau Attention (2015): Pre-Attention

*Reference: [ChatGPT conversation on Bahdanau Attention](https://chatgpt.com/s/t_6a5e627e21948191948ac92a91546104)*

### Expressiveness — The Most Expressive Score Function

*Reference: [ChatGPT conversation on expressiveness](https://chatgpt.com/s/t_6a5e6397c48081918fce9caee257fd5b)*

In attention mechanisms, **expressiveness** means how flexible and powerful the score function is at learning complex relationships between the decoder hidden state and the encoder hidden state.

- A more expressive score function can learn more complicated matching patterns.
- A less expressive one is limited to simpler comparisons.

**Among common attention score functions, Additive (Bahdanau) Attention is the most expressive**, because it uses learnable transformations and a nonlinear activation (`tanh`), allowing it to model more complex alignments than Dot Product or Bilinear attention.

---

## What Does Attention Learn? — Alignment

After training, the attention mechanism automatically learns **word alignments** between the source and target sentences — even though it's never explicitly told which words correspond to each other.

- By visualizing the attention weights `α_tk` as a **heatmap**, each decoder word attends most strongly to the relevant source word, producing bright cells at the aligned positions.

### What Attention Gives Us

*Reference: [ChatGPT conversation on what attention gives us](https://chatgpt.com/s/t_6a5e6623b7fc8191b06e4147f0f90a49)*

### Issues That Remain

*Reference: [ChatGPT conversation on remaining issues](https://chatgpt.com/s/t_6a5e66a1cb3481919382b5094025819c)*
