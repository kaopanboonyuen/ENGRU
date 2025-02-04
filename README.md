# 🚀 ENGRU: Enhanced Gated Recurrent Units for Formal Verification of Colored Petri Nets

![GitHub](https://img.shields.io/github/license/your-repo/ENGRU?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-🔥-red?style=flat-square)

## 📌 Overview

ENGRU (**En**hanced **G**ated **R**ecurrent **U**nits) is a deep learning-based approach to formal verification. It integrates model checking, Colored Petri Nets (CPNs), and sequential learning to analyze discrete-time systems at a high level of abstraction.

### 🧠 Key Features
- **Model Checking Integration**: Uses CPN-based state space exploration.
- **Deep Learning**: Utilizes a GRU-Attention model for sequence learning.
- **Automated Goal-State Identification**: Predicts system behaviors without relying on traditional CTL expressions.
- **High Performance**: Achieves superior speed and accuracy compared to classical model checking.

## 📜 Abstract
Recurrent Neural Networks (RNNs) and automata are fundamental tools for modeling computational systems. In this work, we introduce **ENGRU**, a novel deep learning framework for verifying discrete-time systems. ENGRU transforms CPN-generated state-space graphs into sequential representations, enabling it to predict system behaviors effectively. By leveraging gated recurrent mechanisms, ENGRU enhances goal-state identification, outperforming traditional CTL-based verification.

## 📂 Project Structure
```
ENGRU/
├── dataset/                  # Input dataset files
├── models/                   # Pre-trained model files
├── src/                      # Source code
│   ├── train.py              # Training script
│   ├── dataset_loader.py     # Data preprocessing
│   ├── model.py              # GRU-Attention architecture
│   ├── utils.py              # Helper functions
├── requirements.txt          # Required dependencies
├── README.md                 # Project documentation
```

## 🛠 Installation
```bash
# Clone the repository
git clone https://github.com/kaopanboonyuen/ENGRU.git
cd ENGRU

# Create a virtual environment
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
### 🔹 Training the Model
```bash
python src/train.py
```

### 🔹 Running Inference
```bash
python src/infer.py --input sample_data.txt
```

## 🔬 Model Architecture
```python
class GRUAttentionModel(nn.Module):
    def __init__(self, vocab_size, trans_size, embed_dim=64, hidden_dim=128, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc_stage = nn.Linear(hidden_dim, vocab_size)
        self.fc_trans = nn.Linear(hidden_dim, trans_size)
    
    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        attn_out = attn_out.mean(dim=1)
        return self.fc_stage(attn_out), self.fc_trans(attn_out)
```

## 📊 Experimental Results
| Model | Accuracy | Speedup |
|--------|---------|---------|
| ENGRU | **65.08%** | **3.2x faster** |
| CTL-Based | 48.72% | Baseline |

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve ENGRU!

## 📜 License
This project is licensed under the MIT License.

---

👨‍💻 **Developed by Kao Panboonyuen** 🚀