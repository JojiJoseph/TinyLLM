import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        B, T, D = x.size()

        # Step 1: Project inputs
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)  # (B, T, D)
        V = self.v_proj(x)  # (B, T, D)
        # print("x shape:", x.shape)  # Debugging line
        # print("Q shape:", Q.shape)

        # Step 2: Reshape into heads
        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        
        # print("Before split heads - Q shape:", Q.shape)  # Debugging line
        Q = split_heads(Q)
        # print("After split heads - Q shape:", Q.shape)
        K = split_heads(K)
        V = split_heads(V)

        # Step 3: Compute scaled dot-product attention
        if attn_mask is None:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, H, T, T)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5) + attn_mask.unsqueeze(0).unsqueeze(0)
            
        weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)
        # print("Scores shape:", scores.shape)
        # print("Weights shape:", weights.shape)
        attn_output = torch.matmul(weights, V)  # (B, H, T, d_head)

        # Step 4: Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # Step 5: Final projection
        return self.out_proj(attn_output)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
        

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention + Add & Norm
        B, T, D = x.size()
        attn_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        attn_mask = attn_mask.to(x.device)
        attn_out = self.attn(x, attn_mask=attn_mask)  # (B, T, D)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        # Feed Forward + Add & Norm
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)
        return x

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_length = 10
    d_model = 64
    num_heads = 8

    transformer_block = TransformerBlock(d_model, num_heads, d_ff=d_model)
    input_tensor = torch.randn(batch_size, seq_length, d_model)
    print("Input shape:", input_tensor.shape)  # Should be (B, T, D)

    output_tensor = transformer_block(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be (B, T, D)
