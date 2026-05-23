import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Explicit Self-Attention scoring mechanism mapping sequence hidden states to
    a context-aware representation using scaled dot-product attention.
    Applies a causal mask to prevent looking ahead into the future.
    """
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = 1.0 / (attention_dim ** 0.5)
        self.out_proj = nn.Linear(attention_dim, input_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        Q = self.query(x)  # [batch_size, seq_len, attention_dim]
        K = self.key(x)    # [batch_size, seq_len, attention_dim]
        V = self.value(x)  # [batch_size, seq_len, attention_dim]
        
        # Calculate attention scores (scaled dot product)
        # scores shape: [batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply causal mask (look-ahead mask)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -float('inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted combination of values
        context = torch.matmul(attn_weights, V)  # [batch_size, seq_len, attention_dim]
        output = self.out_proj(context)          # [batch_size, seq_len, input_dim]
        
        return output, attn_weights

class AttentionLSTM(nn.Module):
    """
    Attention-LSTM network featuring:
    - Stacked Uni-directional LSTM layers (causal temporal context)
    - Explicit Causal Self-Attention layer with residual connection
    - Dropout layer
    - Linear projection layer to the vocabulary size
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, attention_dim=128, dropout=0.4):
        super(AttentionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Stacked Uni-directional LSTM for causal sequence modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Uni-directional LSTM output size is hidden_dim
        lstm_output_dim = hidden_dim
        
        # Explicit Causal Self-Attention layer
        self.attention = SelfAttention(lstm_output_dim, attention_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, vocab_size)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, lstm_output_dim]
        
        # Apply dropout to LSTM outputs
        lstm_out = self.dropout(lstm_out)
        
        # Self-attention scoring with causal mask
        attn_out, attn_weights = self.attention(lstm_out)
        
        # Residual connection
        combined = lstm_out + attn_out  # [batch_size, seq_len, lstm_output_dim]
        
        # Apply dropout before projecting to vocab size
        out = self.dropout(combined)  # [batch_size, seq_len, lstm_output_dim]
        logits = self.fc(out)  # [batch_size, seq_len, vocab_size]
        
        return logits
