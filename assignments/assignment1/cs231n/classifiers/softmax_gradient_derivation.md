# Softmax Gradient Derivation

This document explains the mathematical derivation behind the gradient update line in `softmax_loss_naive`:

```python
dW[:, j] += X[i] * dscore
```

---

## 1. Notation and Setup

| Symbol | Shape | Description |
|--------|-------|-------------|
| $N$ | scalar | Number of training examples |
| $D$ | scalar | Input dimension (number of features) |
| $C$ | scalar | Number of classes |
| $\mathbf{X}$ | $(N, D)$ | Input data matrix |
| $\mathbf{x}_i$ | $(D,)$ | The $i$-th training example (row of $\mathbf{X}$) |
| $\mathbf{W}$ | $(D, C)$ | Weight matrix |
| $\mathbf{y}$ | $(N,)$ | Labels, where $y_i \in \{0, 1, ..., C-1\}$ |

---

## 2. Forward Pass

### Step 1: Compute Raw Scores (Logits)

For the $i$-th training example and class $j$:

$$z_j = \mathbf{x}_i \cdot \mathbf{W}_{:,j} = \sum_{k=0}^{D-1} x_{i,k} \cdot W_{k,j}$$

In code: `scores = X[i].dot(W)` → shape $(C,)$

### Step 2: Compute Softmax Probabilities

$$p_j = \frac{e^{z_j}}{\sum_{c=0}^{C-1} e^{z_c}}$$

In code: 
```python
scores -= np.max(scores)  # numerical stability
p = np.exp(scores)
p /= p.sum()
```

### Step 3: Compute Cross-Entropy Loss

$$L_i = -\log(p_{y_i})$$

This is the negative log probability of the correct class.

---

## 3. Backward Pass (Gradient Derivation)

We want to compute $\frac{\partial L_i}{\partial W_{k,j}}$ for all weights.

### Using the Chain Rule

$$\frac{\partial L_i}{\partial W_{k,j}} = \frac{\partial L_i}{\partial z_j} \cdot \frac{\partial z_j}{\partial W_{k,j}}$$

Let's compute each term separately.

---

### Term 1: $\frac{\partial L_i}{\partial z_j}$ (How loss changes w.r.t. score)

This is a well-known result for softmax + cross-entropy:

$$\frac{\partial L_i}{\partial z_j} = 
\begin{cases} 
p_j - 1 & \text{if } j = y_i \text{ (correct class)} \\
p_j & \text{if } j \neq y_i \text{ (wrong class)}
\end{cases}$$

**This is `dscore` in the code!**

```python
if j == y[i]:
    dscore = p[j] - 1
else:
    dscore = p[j]
```

#### Why is this the case?

<details>
<summary>Click to expand full derivation</summary>

The loss is:
$$L_i = -\log(p_{y_i}) = -\log\left(\frac{e^{z_{y_i}}}{\sum_c e^{z_c}}\right) = -z_{y_i} + \log\left(\sum_c e^{z_c}\right)$$

**Case 1: $j = y_i$ (correct class)**

$$\frac{\partial L_i}{\partial z_j} = -1 + \frac{e^{z_j}}{\sum_c e^{z_c}} = -1 + p_j = p_j - 1$$

**Case 2: $j \neq y_i$ (wrong class)**

$$\frac{\partial L_i}{\partial z_j} = 0 + \frac{e^{z_j}}{\sum_c e^{z_c}} = p_j$$

</details>

---

### Term 2: $\frac{\partial z_j}{\partial W_{k,j}}$ (How score changes w.r.t. weight)

Recall:
$$z_j = \sum_{k=0}^{D-1} x_{i,k} \cdot W_{k,j}$$

Taking the partial derivative with respect to $W_{k,j}$:

$$\frac{\partial z_j}{\partial W_{k,j}} = x_{i,k}$$

**This is why we multiply by `X[i]` in the code!**

---

### Combining Both Terms

$$\frac{\partial L_i}{\partial W_{k,j}} = \underbrace{\frac{\partial L_i}{\partial z_j}}_{\text{dscore}} \cdot \underbrace{\frac{\partial z_j}{\partial W_{k,j}}}_{x_{i,k}}$$

For the entire column $j$ of the weight matrix:

$$\frac{\partial L_i}{\partial \mathbf{W}_{:,j}} = \mathbf{x}_i \cdot \text{dscore}$$

In code:
```python
dW[:, j] += X[i] * dscore
```

---

## 4. Dimension Analysis

| Expression | Shape | Description |
|------------|-------|-------------|
| `dW` | $(D, C)$ | Gradient of loss w.r.t. weights |
| `dW[:, j]` | $(D,)$ | Gradient for the $j$-th class column |
| `X[i]` | $(D,)$ | Feature vector of $i$-th example |
| `dscore` | scalar | $\frac{\partial L_i}{\partial z_j}$ |
| `X[i] * dscore` | $(D,)$ | Scalar broadcast multiplication |

---

## 5. Intuition

The gradient update `dW[:, j] += X[i] * dscore` says:

1. **`X[i]`** — Which input features were active for this example?
2. **`dscore`** — How wrong was our prediction for class $j$?
3. **Product** — Adjust weights proportionally to both

### Key Insights:

- If $j$ is the **correct class** ($j = y_i$):
  - `dscore = p[j] - 1` is **negative** (since $0 < p_j < 1$)
  - The gradient will **increase** $W_{:,j}$ to make the correct class score higher

- If $j$ is a **wrong class** ($j \neq y_i$):
  - `dscore = p[j]` is **positive**
  - The gradient will **decrease** $W_{:,j}$ to make this wrong class score lower

---

## 6. Summary

The line:
```python
dW[:, j] += X[i] * dscore
```

Is a direct application of the **chain rule**:

$$\boxed{\frac{\partial L}{\partial \mathbf{W}_{:,j}} = \mathbf{x}_i \cdot (p_j - \mathbf{1}_{j=y_i})}$$

Where $\mathbf{1}_{j=y_i}$ is an indicator function that equals 1 when $j$ is the correct class.
