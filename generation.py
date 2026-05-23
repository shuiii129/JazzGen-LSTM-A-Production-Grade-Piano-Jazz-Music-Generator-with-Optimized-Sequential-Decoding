import torch
import torch.nn.functional as F
import random

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filters a distribution of logits using top-k and/or nucleus (top-p) filtering.
    """
    assert logits.dim() == 1  # Expects a 1D tensor for single sequence generation step
    
    if top_k > 0:
        # Keep only top k logits
        top_k = min(top_k, logits.size(-1))
        # Find the threshold of the top k-th value
        threshold = torch.topk(logits, top_k)[0][-1]
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Identify indices to remove (cumulative probability exceeds top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep the first token exceeding top_p
        # (similar to HuggingFace implementation)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Map sorted indices to remove back to original positions
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits

def generate_sequence(
    model,
    seed_indices,
    num_generate,
    sequence_length=100,
    temperature=1.0,
    top_k=5,
    top_p=0.0,
    repetition_penalty=1.2,
    penalty_window=10,
    jitter_method="second",
    device="cpu"
):
    """
    Generates a sequence of token indices autoregressively from the model.
    Avoids sampling special tokens (<PAD> and <UNK>).
    Forces Temperature Scaling in [0.8, 1.2] and Top-k sampling, unless Top-p is active.
    """
    model.eval()
    generated = list(seed_indices)
    
    # Force Temperature Scaling to be within [0.8, 1.2]
    clamped_temp = max(0.8, min(1.2, temperature))
    if clamped_temp != temperature:
        print(f"  [Inference] Temperature clamped from {temperature} to {clamped_temp}")
        
    # Enforce Top-p bounds: if top_p > 0.0, clamp between [0.85, 0.92]
    effective_top_p = top_p
    if top_p > 0.0:
        effective_top_p = max(0.85, min(0.92, top_p))
        if effective_top_p != top_p:
            print(f"  [Inference] Top-p clamped from {top_p} to {effective_top_p}")
            
    # Top-k: if top_p is active, we can default top_k to 0 to use Top-p exclusively.
    # If top_p is not active (0.0), then we force Top-k to be positive (default 5).
    if effective_top_p > 0.0:
        effective_top_k = top_k  # Allow user choice, but if they pass top_k=0, top-k is disabled.
    else:
        effective_top_k = top_k if top_k > 0 else 5
        if effective_top_k != top_k:
            print(f"  [Inference] Top-k enforced to {effective_top_k}")
            
    with torch.no_grad():
        for _ in range(num_generate):
            # Select/slice context to fit sequence_length
            if len(generated) >= sequence_length:
                input_seq = generated[-sequence_length:]
            else:
                # Pad with 0 (<PAD>) at the beginning
                padding = [0] * (sequence_length - len(generated))
                input_seq = padding + generated
                
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)  # Shape: [1, seq_len]
            
            # Forward pass
            logits = model(input_tensor)  # Shape: [1, seq_len, vocab_size]
            logits = logits[0, -1, :]  # Shape: [vocab_size]
            
            # Explicitly disallow special tokens <PAD> (0) and <UNK> (1) from being generated
            logits[0] = -float('Inf')
            logits[1] = -float('Inf')
            
            # 1. Random Seed Jittering / Stuck Loop Breaking
            # Check if last 3 generated tokens are identical
            if len(generated) >= 3 and generated[-1] == generated[-2] == generated[-3]:
                stuck_token = generated[-1]
                print(f"  [Inference] Stuck loop detected for token {stuck_token}! Injecting jitter: {jitter_method}")
                
                if jitter_method == "second":
                    # Pick 2nd most probable token. We set stuck token logit to -inf
                    # and take the argmax of the remaining.
                    logits[stuck_token] = -float('Inf')
                    next_token = torch.argmax(logits).item()
                    generated.append(next_token)
                    continue
                elif jitter_method == "random":
                    # Sample a completely random token from valid non-special vocabulary (excluding stuck_token)
                    vocab_size = logits.size(-1)
                    valid_indices = [i for i in range(vocab_size) if i not in [0, 1, stuck_token]]
                    if valid_indices:
                        next_token = random.choice(valid_indices)
                    else:
                        next_token = random.choice(range(vocab_size))
                    generated.append(next_token)
                    continue
            
            # 2. Repetition Penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                # Get window of past tokens
                window_start = max(0, len(generated) - penalty_window)
                window_tokens = generated[window_start:]
                
                # Count frequencies in window
                token_counts = {}
                for t in window_tokens:
                    token_counts[t] = token_counts.get(t, 0) + 1
                
                # Apply penalty: logits[t] = logit_val / (repetition_penalty ** count) if logit_val > 0 else logit_val * (repetition_penalty ** count)
                for t, count in token_counts.items():
                    # Skip special tokens if they are in window (unlikely but safe)
                    if t in [0, 1]:
                        continue
                    logit_val = logits[t].item()
                    penalty_factor = repetition_penalty ** count
                    if logit_val > 0:
                        logits[t] = logit_val / penalty_factor
                    else:
                        logits[t] = logit_val * penalty_factor
            
            # Apply temperature scaling
            logits = logits / clamped_temp
                
            # Apply Top-k/Top-p filtering
            filtered_logits = top_k_top_p_filtering(logits, top_k=effective_top_k, top_p=effective_top_p)
            
            # Calculate probabilities
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            generated.append(next_token)
            
    return generated

