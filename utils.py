import torch
import torch.nn.functional as F

MAX_LEN = 256

def sample_next_tokens(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
    """
    Generated using ChatGPT
    Autoregressively samples next tokens from the model given a prompt.
    
    Args:
        model: Your LLM (GPT-style model).
        tokenizer: GPT2 tokenizer.
        prompt (str): Starting text prompt.
        max_new_tokens (int): Number of tokens to generate.
        temperature (float): Sampling temperature (>1 is more random, <1 is more greedy).
        top_k (int): Top-k sampling to filter out low-probability tokens.
    
    Returns:
        str: Generated text (prompt + sampled tokens).
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)  # shape (1, T)
    
    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= MAX_LEN:
            input_ids = input_ids[:, -MAX_LEN:]  # keep only recent tokens

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)  # (1, T, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Top-k filtering
            topk_logits, topk_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            next_token = topk_indices[0, torch.multinomial(probs, num_samples=1)]
            # print(f"Next token ID: {next_token}")
            # print("Input IDs shape:", input_ids.shape)

            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode to text
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text
