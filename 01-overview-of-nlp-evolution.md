# Word Embeddings, Sequence Models, Attention, Transformers & Scaling Laws

## Dense Word Embeddings (Era 3)

In Era 3, a word is represented as a **dense vector** (embedding) — a long list of numbers, e.g. `[0.12, -0.54, 0.89, ...]`.

- Every word is mapped to a specific point in a "meaning space" using these numbers as coordinates. Since words are now numerical, the computer can measure the **distance** and **direction** between them.
- Every entry in the vector acts as a coordinate along a specific axis. A 300-value vector means the word lives at a unique point in a 300-dimensional space.
- These values are **not hand-assigned** — they are **learned** by the machine through training on massive amounts of text.

### Dense vs. Sparse
Called "dense" because almost every entry in the list has a non-zero value, capturing complex, nuanced information about how a word is used in language (as opposed to sparse representations like one-hot vectors).

## How We Train Dense Semantic Vectors

We train dense semantic vectors by forcing a simple **three-layer neural network** to play a predictive "fill-in-the-blank" game (e.g., Google's **Word2Vec**) using a sliding window across a massive text dataset.

- **Skip-gram**: takes a single target word and tries to predict its surrounding context words.
- **CBOW (Continuous Bag of Words)**: does the opposite — uses surrounding words to predict a missing target word.

### Training Process
1. At the start, all words are assigned vectors of completely random numbers, scattered chaotically across an n-dimensional space.
2. As the network repeatedly guesses contexts, it calculates errors and uses **backpropagation (gradient descent)** to continuously tweak the weights of its bottleneck hidden layer — which acts as the coordinate map.
3. **Negative Sampling** is used as a shortcut to make the immense vocabulary calculation computationally practical — it forces the network to distinguish the true context word from just a handful of randomly chosen incorrect words, rather than checking the entire dictionary.
4. After millions of iterations, words that share similar real-world contexts are nudged into the same neighborhood of the vector space.
5. The predictive outer layers are discarded, leaving behind the perfectly clustered **dense semantic vectors** as a byproduct.

## Sequence Models

### Sequence-to-Sequence (Seq2Seq)

Introduced in 2014 by researchers at Google (Sutskever et al.) and Cho et al. — a massive breakthrough in Era 3 designed to map an input sequence of variable length to an output sequence of variable length.

- Before Seq2Seq, neural networks required **fixed-size** inputs and outputs (like classifying a fixed 28x28 pixel image).
- Seq2Seq solved this using an **Encoder-Decoder architecture**.

```
INPUT: "The cat sat" ──> [ ENCODER ] ──> ( Context Vector ) ──> [ DECODER ] ──> OUTPUT: "Le chat s'est"
```

**Became the underlying engine for:**
- **Machine Translation** (powering Google Translate's massive quality leap in 2016)
- **Text Summarization** (reading a long article and generating a short summary)
- **Early Chatbots & Conversational AI** (taking a user question as input, generating an answer as output)

## Attention Mechanism

The Attention Mechanism was explicitly designed to give **dynamic weights** (importance scores) to different words in the input sequence, depending on what the model was trying to generate at that exact moment.

- Instead of forcing the model to rely on one single, heavily compressed "Context Vector" for the entire sentence, attention gave the Decoder a mathematical "magnifying glass" to look back at the original input.

## Transformers

Transformers introduced **Self-Attention**, allowing the model to ingest an entire paragraph — or an entire book — all at once, in parallel.

- Instead of waiting in a sequential line, every word in a document looks at every other word simultaneously.
- This unlocked the raw power of GPUs, allowing AI training to scale up by thousands of times.

### Self-Attention in Action

Transformers use Self-Attention to allow a word to look at its surrounding sentence to determine its exact meaning before doing any translation or generation.

Example — the word **"bank"**:
- *"The robber ran out of the money bank."*
- *"The fisherman sat on the river bank."*

In older Era 3 models (like Word2Vec), "bank" had only **one fixed static vector** — it couldn't change.

The Transformer fixes this:
- In sentence 1, Self-Attention assigns high weights to "robber" and "money," dynamically shifting the vector of "bank" closer to the "financial" neighborhood.
- In sentence 2, it weights "fisherman" and "river," shifting the same word "bank" into the "nature" neighborhood.
- This creates **dynamic, context-aware vectors**.

### Attention (as a Helper) vs. Transformer (as an Engine)

| | Role |
|---|---|
| **Attention Mechanism** | A "helper tool" added to old, slow networks (RNNs) to stop them from forgetting words in long sentences — but the model still read text slowly, one word at a time |
| **Transformer** | A brand-new engine that threw away old networks entirely and built the whole system around attention |

- The Transformer looks at an entire paragraph of text at once (in parallel), and uses Self-Attention to let words look at their neighbors to figure out their exact meaning based on context (e.g., knowing if "bank" means a riverbank or a money bank).
- In short: attention was an accessory; the Transformer turned that accessory into a super-fast, context-smart standalone engine that made modern AI possible.
- Using Self-Attention, words look at the **whole text** to decide their meaning, rather than relying on a fixed, pre-determined "closest word" in space. Older systems forced words into fixed positions based on general similarity; Self-Attention gives words the power to dynamically adapt their position by evaluating the entire context they're currently sitting in.

## Why Transformers Won

Transformers won the AI race because they solved the single biggest bottleneck in computer science: **scalability**. Older networks were smart, but too slow to train on massive amounts of data.

### Hardware Efficiency (Parallel Processing)
- Older models (LSTMs and RNNs) had to read text **sequentially** — word by word. If a computer had 1,000 powerful processors, 999 sat idle waiting for the first processor to finish reading the first word.
- Transformers process the entire text **all at once**, letting researchers hook up thousands of high-powered GPUs to calculate billions of word relationships simultaneously.

## The Scaling Laws Story

### The Kaplan Laws (OpenAI, 2020)

- **Making the model bigger** = "Increasing Brain Capacity"
- **Giving it more data** = "Giving it More Textbooks to Read"

A brilliant AI cannot exist without a perfect balance of both size and data:

- **Big Brain + Very Little Data**: A trillion-parameter model trained on only a few hundred books becomes an "undertrained giant."
- **Small Brain + Massive Data**: A tiny model with only a few million parameters, given the entire internet to read, hits an information bottleneck.

**Kaplan's conclusion**: Model size was the absolute king. If you increase your budget, put roughly **73%** into making the model bigger and only **27%** into more data.

### Chinchilla (DeepMind)

- For every single parameter in a model, it must be trained on **at least 20 tokens** of data.
- The Chinchilla correction altered the trajectory of AI development — proving smaller, data-rich models were not only smarter but also exponentially faster and cheaper to run on user devices.
- This insight paved the way for efficient modern architectures like Meta's open-source **LLaMA** series.
- Chinchilla outperformed Kaplan-style models by challenging the "just get bigger" rule — proving smaller models trained on massive data yield better performance and save compute, completely changing how AI is built today.
- **Rule of thumb**: train on approximately 20 tokens per parameter.

### Current Practice — Beyond Chinchilla

- Modern AI training has shifted beyond Chinchilla-optimal points to prioritize **inference efficiency** over training efficiency — using massive datasets for smaller models to reduce lifetime deployment costs.
- Examples: **Llama 3 8B** and the extreme **60,000:1** token-to-parameter ratio of **Qwen3-0.6B** demonstrate this shift toward compact, high-speed models.
- This approach faces diminishing returns and looming data constraints.
- The industry is transitioning toward:
  - **Test-time scaling** — allowing models to boost performance during inference
  - **Synthetic data generation**

### 2024 — Inference-Aware Scaling

> "It depends on your deployment:
> - More Inferences → Smaller models
> - Less Inferences → Bigger Model"
