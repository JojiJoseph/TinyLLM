from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")
# print(ds.head())

from transformers import GPT2Tokenizer
from attention import TransformerBlock

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer("The cat sat on the mat", return_tensors='pt')
print(tokens)
import torch

HIDDEN_DIM = 512
VOCAB_SIZE = tokenizer.vocab_size
MAX_LEN = 256  # Define a maximum sequence length for the model
import torch.nn as nn

class LLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(LLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, MAX_LEN, hidden_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads=8, d_ff=hidden_dim) for _ in range(6)
        ])
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        print("Input shape:", x.shape)
        B, T = x.size()
        x = self.embedding(x) + self.pos_embedding[:, :T]
        print("After embedding shape:", x.shape)
        # x = x.transpose(0, 1)
        for block in self.transformer_blocks:
            x = block(x)
        # x = x.transpose(0, 1)
        x = self.fc(x)
        return x


llm = LLM(VOCAB_SIZE, HIDDEN_DIM).cuda()
x = tokens['input_ids']
x = x.to(torch.int64).cuda()  # Ensure input is on the same device as the model
x = llm(x)
print(x.shape)  # Should be (batch_size, sequence_length, vocab_size)
