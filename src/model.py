import torch
import torch.nn as nn
import torch.nn.functional as F

class ENGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False, dropout=0.1, num_heads=4):
        super(ENGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        
        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        
        # Multi-Head Self-Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * (2 if bidirectional else 1), 
                                               num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout and Layer Normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        
    def forward(self, x):
        # GRU Forward Pass
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_dim * num_directions]
        
        # Apply Layer Normalization
        norm_out = self.layer_norm(gru_out)
        
        # Self-Attention Mechanism
        attn_out, _ = self.attention(norm_out, norm_out, norm_out)  # [batch, seq_len, hidden_dim * num_directions]
        
        # Residual Connection
        attn_out = attn_out + norm_out
        
        # Fully Connected Layers
        x = self.fc1(attn_out)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, seq_len, output_dim]
        
        return x