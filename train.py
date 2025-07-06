from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from models import LLM
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from utils import sample_next_tokens
from tqdm import tqdm


MAX_LEN = 256  # Define a maximum sequence length for the model

ds = load_dataset("roneneldan/TinyStories", split="train")
ds = ds.shuffle(seed=42).select(range(10_000))  # Adjust the range as needed
print(ds[0])  # Example output: {'text': 'Once upon a time...', 'id': 0}


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token

def tokenize(batch):
    return tokenizer(batch["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)


# Tokenize dataset
tokenized_ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=MAX_LEN), batched=True)
tokenized_ds.set_format(type='torch', columns=['input_ids'])

from torch.utils.data import DataLoader

dataloader = DataLoader(tokenized_ds, batch_size=32, shuffle=True)
# Initialize the model
HIDDEN_DIM = 512
VOCAB_SIZE = tokenizer.vocab_size
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

llm = LLM(VOCAB_SIZE, HIDDEN_DIM).cuda()

# Training loop
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(llm.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):  # Adjust the number of epochs as needed
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        input_ids = batch['input_ids'].cuda()
        # print(f"Batch input shape: {input_ids.shape}")  # Debugging line to check input shape
        inputs = input_ids[..., :-1]# .clone()
        targets = input_ids[..., 1:]  # Shifted input for next token prediction
        optimizer.zero_grad()
        
        # Forward pass
        outputs = llm(inputs)
        
        # Compute loss (using CrossEntropyLoss)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, VOCAB_SIZE), targets.reshape(-1))
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(llm.state_dict(), f"llm_epoch{epoch+1}.pt")


llm.load_state_dict(torch.load("llm_epoch1.pt"))



prompt = "Once upon a time, "
generated_text = sample_next_tokens(llm, tokenizer, prompt)
print(generated_text)
