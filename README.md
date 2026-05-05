# BTHA
Block-then-Hash Attention for Efficient Long-Context

🚀 An efficient attention mechanism designed for long-context modeling, combining **block partitioning** and **hash-based attention** to reduce computational complexity while preserving performance.

---

## 📌 Overview

Traditional self-attention mechanisms scale quadratically with sequence length, making them inefficient for long-context tasks.

**BTHA (Block-then-Hash Attention)** addresses this issue by:

1. **Block Partitioning**: Dividing the input sequence into manageable blocks  
2. **Hash-Based Attention**: Applying locality-sensitive hashing (LSH) to approximate global attention  
3. **Hybrid Strategy**: Combining local and global attention efficiently  

This results in:
- ⚡ Reduced computational cost
- 📉 Lower memory usage
- 📈 Strong performance on long sequences

---

## 🧠 Method

### Step 1: Block Partitioning
The input sequence is divided into fixed-size blocks:

```

Sequence → [Block₁, Block₂, ..., Blockₙ]

```

### Step 2: Intra-block Attention
Self-attention is applied within each block:

```

Attention(Blockᵢ)

```

### Step 3: Hash-Based Grouping
Tokens are projected into hash buckets using LSH:

```

Hash(Q, K) → Buckets

```

### Step 4: Inter-block Attention
Attention is computed across tokens within the same hash bucket:

```

Attention(Bucketⱼ)

```

---

## 🏗️ Architecture

```

Input Sequence
↓
Block Partition
↓
Local Attention (within blocks)
↓
Hash Projection (LSH)
↓
Global Attention (within buckets)
↓
Output Representation

````

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/BTHA.git
cd BTHA
pip install -r requirements.txt
````

---

## 🚀 Usage

### Basic Example

```python
from btha.attention import BTHAAttention
import torch

model = BTHAAttention(
    dim=512,
    num_heads=8,
    block_size=64,
    num_hashes=4
)

x = torch.randn(1, 4096, 512)  # long sequence
output = model(x)

print(output.shape)
```

---

## 📊 Complexity

| Method         | Time Complexity | Memory |
| -------------- | --------------- | ------ |
| Full Attention | O(n²)           | High   |
| BTHA           | ~O(n log n)     | Lower  |

---

## 🔬 Experiments

BTHA has been evaluated on:

* Long Document Classification
* Language Modeling
* Sequence Prediction Tasks

### Results Summary

| Model       | Sequence Length | Accuracy   | Speed  |
| ----------- | --------------- | ---------- | ------ |
| Transformer | 4K              | High       | Slow   |
| BTHA        | 16K+            | Comparable | Faster |

---

## 📁 Project Structure

```
BTHA/
│── btha/
│   ├── attention.py
│   ├── hashing.py
│   ├── blocks.py
│── experiments/
│── configs/
│── README.md
│── requirements.txt
```

---

## 🧩 Key Features

* ✅ Scalable to long sequences (10K+ tokens)
* ✅ Efficient memory usage
* ✅ Easy integration with PyTorch
* ✅ Modular design

---

## 📌 Future Work

* Adaptive block sizing
* Better hash functions
* Integration with large language models (LLMs)

---

## 🤝 Contributing

Contributions are welcome!

```bash
fork → clone → branch → commit → PR
```

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

Inspired by prior work on:

* Sparse Attention
* Locality Sensitive Hashing (LSH)
* Efficient Transformers

---

## 📬 Contact

If you have any questions, feel free to open an issue or contact:

* Email: 
* GitHub: (https://github.com/Rekio-ll)
