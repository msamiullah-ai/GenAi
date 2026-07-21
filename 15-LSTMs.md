# LSTM: Long Short-Term Memory

---

## Why LSTM? Recap of the Core Problem

One of the main reasons vanilla RNNs struggle with long-term dependencies: the **same** hidden state (context vector) is forced to perform two different tasks simultaneously:

1. Store information needed for the **current** prediction (short-term context).
2. Preserve important information for **future** predictions (long-term context).

This causes old information to get overwritten or washed away quickly.

---

## LSTM's Fix: Separating the Two Responsibilities

LSTM splits this single overloaded hidden state into **two** distinct components:

### Cell State (`C_t`) → Long-Term Memory
- Stores important information that may be needed many time steps later.
- Carries this information forward across the sequence, largely undisturbed.

### Hidden State (`h_t`) → Short-Term Memory
- Contains the information needed to make the prediction at the **current** time step.
- This is the actual output of the LSTM cell.

---

<img width="761" height="407" alt="image" src="https://github.com/user-attachments/assets/218f6d7d-8419-4d21-ab2f-86c206ac4620" />

---

## The Context Management Problem

Once LSTM has a separate long-term memory, it must answer **two questions** at every time step:

1. **What should be forgotten?** — remove information that's no longer useful. Handled by the **Forget Gate**.

2. **What should be remembered?** — store new important information for future use. Handled by the **Input Gate** (together with the candidate memory).

> 💡 **Important**: these decisions are **not** programmed manually — the gates learn them automatically during training, using Backpropagation Through Time (BPTT). The LSTM makes predictions → loss is computed → backpropagation updates the gate weights → over many examples, the gates learn *when* to keep, forget, or expose information, because those choices reduce prediction error.

---

## Gates — The General Mechanism

A **gate** in an LSTM is a small neural network that produces values between **0 and 1**, using a **sigmoid** activation.

**Inputs to a gate:**
- Current input: `x_t`
- Previous hidden state: `h_{t-1}`

**Computation:**
```
Gate Output = σ(W[h_{t-1}, x_t] + b)
```
where `σ` is the sigmoid function.

*Reference: [LSTM — under the hood](https://chatgpt.com/s/t_6a5dbda49f6081919bac8bb1329cd471)*

### The Three Gates — Overview

| Gate | Question it answers |
|---|---|
| **Forget Gate** | What old information should I remove from the cell state? |
| **Input Gate** | What new information should I store in the cell state? |
| **Output Gate** | What part of the cell state should become the hidden state (current output)? |

---

## Complete Flow: How a Gate Actually Works

At each time step, each LSTM gate computes its control signal through the current input vector (`x_t`) and the previous hidden state (`h_{t-1}`) as its inputs.

1. These inputs are passed through a **feed-forward (fully connected) layer**, which computes a linear transformation and produces an intermediate vector:
   ```
   z = W[h_{t-1}, x_t] + b
   ```

2. This vector `z` is passed through a **sigmoid activation**, mapping each element to a value between 0 and 1.

   > 💡 **These values are not new information or memory themselves** — they act purely as **control signals** specifying how much information should be allowed to pass through.

3. Finally, the gate output is multiplied **element-wise** (Hadamard product `⊙`) with the specific vector that gate controls — e.g., the previous cell state (Forget Gate), the candidate memory (Input Gate), or the activated cell state (Output Gate).

   - Gate values close to **1** preserve the corresponding information almost unchanged.
   - Gate values close to **0** suppress or effectively erase it.

This is how each gate **selectively controls the flow of information** through the LSTM.

---

## Forget Gate

*Reference: [Forget Gate deep-dive](https://chatgpt.com/s/t_6a5dc41b4ea481919d735c01607487cf)*

The Forget Gate determines what information from the previous cell state (`C_{t-1}`) should be retained or discarded.

**How it works:**

1. Uses the previous hidden state (`h_{t-1}`) and current input (`x_t`), passed through a feed-forward layer + sigmoid activation, to produce the forget gate output `f_t` — also called the **selector vector**.

2. Each element of `f_t` lies between 0 and 1, representing how much of the corresponding element in the previous cell state should be preserved.

3. The selector vector `f_t` is multiplied **element-wise** with the previous cell state `C_{t-1}`.
   - Values close to 1 → retain the corresponding memory.
   - Values close to 0 → suppress/remove it.

This produces a **filtered version of the old memory** that gets carried forward.

---

## Input Gate

*Reference: [Input Gate deep-dive](https://chatgpt.com/s/t_6a5dc305490c8191bef2e92cc4301907) ⭐ recommended*

The Input Gate updates the long-term memory using **two independent neural networks**:

1. The **Candidate Memory** network generates potential new information (`g_t`), using a **tanh** activation (values between -1 and 1).

2. The **Input Gate** itself generates a selector vector (`i_t`), using a **sigmoid** activation (values between 0 and 1).

**Combining them:**

- The selector vector `i_t` determines how much of the candidate memory `g_t` should actually be stored, via element-wise multiplication → producing the **approved new memory** (`J_t`).

- `J_t` is then **added** to the filtered old memory (`k_t`, the output of the Forget Gate step) to form the **updated cell state** `C_t`.

```
C_t = (f_t ⊙ C_{t-1}) + (i_t ⊙ g_t)
   =  filtered old memory (k_t)  +  approved new memory (J_t)
```

---

## Output Gate

*Reference: [Output Gate deep-dive](https://chatgpt.com/s/t_6a5dc7a37f248191af98d6b244262be4) ⭐ recommended*

The Output Gate determines what information from the updated cell state (`C_t`) should be exposed as the hidden state (`h_t`).

**How it works:**

1. The updated cell state `C_t` is passed through a **tanh** activation to produce the candidate hidden state `q_t` (values between -1 and 1).

2. Independently, the Output Gate uses the previous hidden state (`h_{t-1}`) and current input (`x_t`) to generate a selector vector `o_t`, through a feed-forward layer + sigmoid activation.

3. The selector vector `o_t` is multiplied element-wise with the candidate hidden state `q_t`, producing the new hidden state:
   ```
   h_t = o_t ⊙ q_t
   ```

The hidden state `h_t` is used both to make the **current prediction**, and as the previous hidden state for the **next** time step — while the cell state `C_t` continues to carry the long-term memory forward.

### The Entire LSTM Algorithm

*Reference: [Full worked-through LSTM algorithm](https://chatgpt.com/s/t_6a5dca2df2c08191bc88960e1828dd9d)*

---

## How LSTM Helps Solve the Vanishing Gradient Problem

**The key insight**: LSTM replaces an *uncontrolled* chain of weight-matrix and activation multiplications (as in a vanilla RNN) with a **controlled memory path**.

- Along this path, the gradient is multiplied **only by the forget gate**.
- If the network learns that some information should be **remembered**, it sets the forget gate close to **1**, allowing gradients to flow across many time steps almost unchanged.
- If it learns the information is **no longer useful**, it sets the forget gate close to **0**, intentionally blocking the gradient.

This is why LSTMs can preserve long-term dependencies far better than vanilla RNNs.

*Further reading: [To-the-point explanation](https://chatgpt.com/s/t_6a5dcfdcc3f88191a287764b71d9fea2) | [If still confused, read this](https://chatgpt.com/s/t_6a5dd00d50e8819195c6e97789798878)*

### An Important Caveat: LSTMs Don't Fully Solve Gradient Problems

> ⚠️ **Exploding gradients in LSTMs are typically handled by gradient clipping — not by the architecture itself.** They are NOT handled automatically by the LSTM design.

**On vanishing gradients specifically:**

- The LSTM architecture makes it *possible* to preserve information over many recurrent time steps.
- **Example**: if the forget gate is close to 1, and the input gate is close to 0, the information in the cell state will be preserved over many recurrent time steps.
- For a vanilla RNN, it's much harder to preserve information in hidden states across time steps, since it relies on just a single weight matrix with no selective control.

> 💡 **The honest bottom line**: LSTMs do **not guarantee** there's no vanishing/exploding gradient problem — but they *do* provide the model with a mechanism (the gates) to actually *learn* long-distance dependencies, which vanilla RNNs structurally cannot do at all.

---

## LSTM Variants

*Reference: [Peephole Connections & Coupled Input-Forget Gate (CIFG)](https://chatgpt.com/s/t_6a5dd2a8f6688191a6d9c9b61a1b017d)*

- **Peephole Connections**: a variant where the gates are also allowed to "peek" at the cell state itself when making their decisions, not just the hidden state and input.
- **Coupled Input-Forget Gate (CIFG)**: a variant that ties the input and forget gates together (using `1 − f_t` in place of a separate input gate), reducing the number of parameters.

---

## Gated Recurrent Unit (GRU)

*Reference: [Introduction to GRU](https://chatgpt.com/s/t_6a5dd45c44408191abcec891cd439d17)*

A simpler alternative to LSTM that merges the cell state and hidden state into one, and uses fewer gates — covered in more depth separately.
