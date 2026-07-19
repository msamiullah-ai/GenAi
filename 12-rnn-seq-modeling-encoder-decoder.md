# RNNs, Encoder-Decoder Architecture & Autoregressive Generation

---

## Reference Materials

*Slides: [Neural network figures — full slide deck](https://pern-my.sharepoint.com/:b:/g/personal/namoos_qasmi_lums_edu_pk/IQAaOjeQvHGBQbVQALu3dt6JAeyCLt4aNFOUPWJ2sUyoYZU?e=Yt2pdo)*

*Explanations: [Gemini conversation walking through the figures](https://share.gemini.google/rqDXTBtKD5uh)*

*Complete lecture figures/diagrams explanation, in flow: [Gemini link](https://share.gemini.google/N2MDoICHCflj) and the [slides link above](https://pern-my.sharepoint.com/:b:/g/personal/namoos_qasmi_lums_edu_pk/IQAaOjeQvHGBQbVQALu3dt6JAeyCLt4aNFOUPWJ2sUyoYZU?e=Yt2pdo)*

### Figure Index (for reference against the slide deck)

| Figure | Title |
|---|---|
| 1 (3-word context model) | Bengio-Style Feedforward Neural Network (FFNN) Language Model |
| 2 (isolated green blocks layout) | Limitations of Standard Feedforward Networks on Sequential Data |
| 3 (red circle inputs introduced) | Conceptual Transition to Recurrent Neural Networks (RNNs) |
| 4 (FFNN Bigram warning model) | Feedforward Neural Network Bigram Language Model (No Memory) |
| 5 ("I like deep" sequential rows) | Isolated Word-by-Word Processing in Standard Feedforward Networks |
| 6 (unrolled multi-row model) | Recurrent Neural Network (RNN) Language Model Unrolled Across Time Steps |
| 7 (single time step matrix layout) | Architecture and Matrix Dimensions of a Single Time Step RNN Layer |

---

## Why RNNs? The Biological Inspiration

Recurrent Neural Networks are inspired by a part of the brain called the **Temporal Lobe**, which is involved in short-term memory.

The main idea behind an RNN is to **mimic short-term memory**. Just as humans remember recent information while reading or listening, an RNN maintains a **hidden state** that stores a summary of the sequence it has seen so far.

- Instead of processing the entire input at once (like an FFNN), it reads **one element at a time**, continuously updating its memory using the current input and the previous hidden state.
- This lets the network capture **short-term dependencies**, where the meaning or prediction of the current word depends on recently seen words.

Because of this ability to remember recent context, RNNs are well suited for sequential tasks: language modeling, speech recognition, machine translation, and time-series analysis.

---

## Parameter Sharing and Memory — The Core Concept

The defining breakthrough of an RNN is elegant and simple: instead of building brand-new network structures for every single word in a sequence, it uses the **exact same set of weights (parameters)** at every time step, and passes forward a hidden state that summarizes the sequence's history.

At every time step `t`, the network applies the **same** mathematical function `f` with the **same** parameters `θ`:

```
h_t = f(h_{t-1}, x_t; θ)
```

This formula states exactly what the diagrams show: the new hidden state (`h_t`) is calculated by combining the past hidden state / memory (`h_{t-1}`) with the current incoming word vector (`x_t`).

> 💡 A **vanilla RNN has only one hidden state per time step**. This state — usually denoted `h_t` — acts as **both** the network's working memory for the sequence **and** its output at that specific moment.

---

## The Limitation: Vanilla RNNs and Information Dilution

A major limitation of a vanilla RNN is that it struggles to remember information over **long sequences**.

- At every time step, the hidden state is updated using the previous hidden state and the current input — meaning the **same memory slot is repeatedly overwritten**.
- Since this update also passes through activation functions like `tanh` or `sigmoid`, which compress values into a limited range, the influence of older information **gradually weakens** as more words are processed.

This phenomenon is called **information dilution** — earlier context becomes increasingly faint and difficult to recover.

> 💡 **Why this happens mechanically**: information gets diluted because the hidden state passes through non-linear activation functions (like `tanh`) at every time step. Since these functions map values into a specific bounded range (e.g., -1 to 1), the values are essentially **scaled down** each time they're processed. As the sequence progresses, the earlier "flavors" or signals are repeatedly multiplied by these compressing values, causing them to become increasingly weaker and more diluted over time.

**Result**: vanilla RNNs capture short-term dependencies effectively, but often fail to preserve **long-term dependencies**, making it difficult to use information from the distant past. This inherent memory bottleneck motivated the development of more advanced architectures like **LSTMs and GRUs**, specifically designed to retain important information over longer sequences.

---

## Encoding: Compressing a Sequence Into a Vector

As an RNN processes a sequence one word at a time, it continuously updates its hidden state by combining the current input with the previous memory. After reading the **final** word, the last hidden state contains a single fixed-size vector that serves as a numerical representation of the **entire** sequence.

- This process is called **encoding**, and the RNN performing this role is known as an **encoder** — it compresses the complete input sequence into one representative vector, often called a **sentence embedding**.

> ⚠️ **This compression is lossy, not lossless.** Since a long sentence must be represented by a fixed-length vector, some information — especially from earlier parts of long sequences — may be weakened or lost due to the RNN's limited memory capacity. This **single-vector bottleneck** became a major limitation of encoder-decoder models, and later motivated the development of **Attention** and **Transformer** architectures, which avoid relying on only one compressed representation.

---

## Why Vanilla (Left-to-Right) RNNs Struggle: A Worked Example

Consider this sentence with a blank:

> *"Because ______ was feeling completely exhausted after the long flight, Sarah went straight to sleep."*

- If you read only up to the blank ("Because ______"), you have **no context** to fill it. It could be "he," "she," "they," "I," or "it."
- However, by looking at what comes *after* the blank, you find the proper noun **"Sarah."** Because "Sarah" appears later in the sequence, you instantly know the blank must be filled with "she" to maintain proper grammatical reference.

**Why vanilla RNNs struggle with this**: as a vanilla RNN processes this sentence word-by-word, it reaches the blank **before** it has ever seen the word "Sarah." Its "smoothie" memory vector at that exact moment only contains the flavor of the word "Because" — it's completely blind to the crucial future context waiting later in the sentence.

### The Fix: Bidirectional RNNs

To solve this limitation, engineers developed **Bidirectional RNNs**, which run two separate "blenders" simultaneously: one moving forward (left-to-right), and another moving backward (right-to-left). This lets the network use **both** past context and future context to fill in the blanks accurately.

> 💡 **Context is genuinely bidirectional** — ambiguity often gets resolved by looking both backward *and* forward through a sentence. This is why the industrial/production-standard version of RNN usage is the **Bidirectional RNN**, where a sentence is processed in both directions. In this setup, the number of embeddings representing a word effectively equals **the context surrounding that word** (from both directions combined).

### Bidirectional RNN (BiRNN) — Formal Definition

*Reference: [ChatGPT conversation on BiRNNs](https://chatgpt.com/s/t_6a5b2b1552248191856959bfc0d11045)*

A BiRNN extends a standard RNN by processing the input sequence in **both** forward and backward directions.

- A normal RNN reads left-to-right and can only use information from **previous** words.
- A BiRNN consists of **two separate RNNs**: one processes the sequence left-to-right (capturing past context), the other right-to-left (capturing future context).
- For each word, the hidden states from both RNNs are combined — usually by **concatenation** — to produce a richer representation containing information from both preceding and succeeding words.

This makes BiRNNs highly effective for tasks like sentiment analysis, named entity recognition, speech recognition, and part-of-speech tagging.

> ⚠️ **Key limitation**: BiRNNs **cannot** be used for tasks like next-word prediction or text generation, because future words are simply not available during inference — there's nothing to "look ahead" to yet.

---

## The Encoder–Decoder Architecture

**The problem it solves**: what if the input sequence and the output sequence don't have the same length?

The **Encoder–Decoder architecture** is designed for sequence-to-sequence tasks where input and output sequences can have different lengths — machine translation, speech recognition, text summarization, question answering.

- Instead of using a single RNN, it uses **two separate RNNs**.
- The **encoder** reads the input sequence one element at a time and compresses its overall meaning into a fixed-size **context vector** (the final hidden state) — a summary of the entire input.
- The **decoder** then takes this context vector as its initial information and generates the output sequence one element at a time, until it predicts an end-of-sequence token.

```
Input Sequence
      │
      ▼
  Encoder
      │
      ▼
 Context Vector
      │
      ▼
  Decoder
      │
      ▼
Output Sequence
```

Although this architecture was a major breakthrough, representing an entire input sequence with a **single** context vector creates a bottleneck — especially for long sequences — motivating later improvements such as the **Attention** mechanism.

### Roles, Not Specific Architectures

- An **encoder** reads the entire input sequence and converts it into numerical representations (hidden states) that capture the meaning of the input. From these hidden states, a context vector is created, summarizing the essential information.
- The **decoder** receives this context vector as its starting point and generates the output sequence one element at a time — producing its own hidden states and converting each hidden state into an output token.

> 💡 **Important clarification**: "Encoder" and "decoder" are **roles**, not specific neural network types. RNNs, LSTMs, CNNs, and Transformers can all be used to implement either role — this is a functional pattern, not tied to any one architecture.

---

## Autoregressive Generation

**Autoregressive generation** (or **causal language model generation**) generates text one word at a time, where each new word is predicted based on **all previously generated words**.

**The process:**

1. Generation starts by feeding the special **start-of-sentence token** `<s>` into the model.

2. The token is converted into its embedding and passed through the RNN, which updates its hidden state and produces scores for every word in the vocabulary.

3. A **Softmax** layer converts these scores into probabilities, and one word is **sampled** from this probability distribution.

4. The sampled word becomes both the next generated word **and** the input for the next time step (after being converted to its embedding) — while the hidden state carries forward the context of all previous words.

This **prediction → Softmax → sampling → feedback loop** repeats until the model generates the **end-of-sentence token** `</s>`, or a predefined maximum length is reached.

> 💡 This simple generation mechanism is the foundation of language generation, machine translation, text summarization, and question answering — the only difference between these applications is the **initial context** provided before generation begins.
