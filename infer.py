import torch
from transformers import GPT2Tokenizer
from models import LLM
from utils import sample_next_tokens
from models import LLM

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a pad token

HIDDEN_DIM = 512
VOCAB_SIZE = tokenizer.vocab_size

llm = LLM(VOCAB_SIZE, HIDDEN_DIM).cuda()

llm.load_state_dict(torch.load("llm_epoch3.pt"))



prompt = "Once upon a time, "
generated_text = sample_next_tokens(llm, tokenizer, prompt)
print(generated_text)
