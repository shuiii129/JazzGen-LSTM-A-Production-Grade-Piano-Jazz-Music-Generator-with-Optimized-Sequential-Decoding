import os
import argparse
import pickle
import random
import torch
from model import AttentionLSTM
from generation import generate_sequence
from export import export_tokens_to_midi
from data_pipeline import parse_midi_to_tokens

def generate(
    checkpoint_path,
    vocab_path,
    output_path,
    seed_midi=None,
    num_generate=200,
    sequence_length=100,
    temperature=1.0,
    top_k=0,
    top_p=0.90,
    repetition_penalty=1.2,
    penalty_window=10,
    jitter_method="second",
    tempo=120,
    humanize_tempo=True
):
    # 1. Load Vocab Mappings
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please train the model first.")
        
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
    token_to_idx = vocab_data["token_to_idx"]
    idx_to_token = vocab_data["idx_to_token"]
    vocab_size = len(token_to_idx)
    
    # 2. Load Model Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please train the model first.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model architecture from checkpoint parameters
    model = AttentionLSTM(
        vocab_size=vocab_size,
        embedding_dim=checkpoint.get('embedding_dim', 128),
        hidden_dim=checkpoint.get('hidden_dim', 256),
        num_layers=checkpoint.get('num_layers', 2),
        attention_dim=checkpoint.get('attention_dim', 128),
        dropout=checkpoint.get('dropout', 0.3)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 0) + 1})")
    
    # 3. Handle Seed Setup
    seed_indices = []
    if seed_midi and os.path.exists(seed_midi):
        print(f"Parsing seed MIDI: {seed_midi}")
        seed_tokens = parse_midi_to_tokens(seed_midi)
        if seed_tokens:
            seed_indices = [token_to_idx.get(t, token_to_idx["<UNK>"]) for t in seed_tokens]
            print(f"Loaded {len(seed_indices)} seed tokens from MIDI.")
            
    if not seed_indices:
        # Fallback to random seed from vocab (excluding special tokens)
        print("No seed MIDI provided or parsing failed. Generating from random seed token...")
        usable_indices = [idx for token, idx in token_to_idx.items() if token not in ["<PAD>", "<UNK>"]]
        if usable_indices:
            random_idx = random.choice(usable_indices)
            seed_indices = [random_idx]
        else:
            seed_indices = [0]
            
    # 4. Generate Sequence
    print("Generating sequence...")
    generated_indices = generate_sequence(
        model=model,
        seed_indices=seed_indices,
        num_generate=num_generate,
        sequence_length=sequence_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        penalty_window=penalty_window,
        jitter_method=jitter_method,
        device=device
    )
    
    # Map indices back to token strings
    generated_tokens = [idx_to_token.get(idx, "<UNK>") for idx in generated_indices]
    
    # 5. Export to MIDI file
    export_tokens_to_midi(
        tokens=generated_tokens,
        output_path=output_path,
        tempo=tempo,
        humanize_tempo=humanize_tempo
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI using trained Attention-LSTM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to best_model.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.pkl", help="Path to vocab.pkl")
    parser.add_argument("--output", type=str, default="output.mid", help="Path to save output MIDI file")
    parser.add_argument("--seed_midi", type=str, default=None, help="Path to seed MIDI file (optional)")
    parser.add_argument("--num_generate", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length model uses")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering (default 0, disabled when top_p is active)")
    parser.add_argument("--top_p", type=float, default=0.90, help="Top-p (nucleus) filtering (e.g. 0.85-0.92, default 0.90)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty factor (default 1.2)")
    parser.add_argument("--penalty_window", type=int, default=10, help="Number of past tokens to look back for repetition penalty (default 10)")
    parser.add_argument("--jitter_method", type=str, choices=["second", "random", "none"], default="second", help="Method to resolve stuck chord loops (default: second)")
    parser.add_argument("--tempo", type=int, default=120, help="Default tempo (BPM)")
    parser.add_argument("--no_humanize_tempo", action="store_true", help="Disable expressive tempo changes")
    
    args = parser.parse_args()
    
    generate(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        output_path=args.output,
        seed_midi=args.seed_midi,
        num_generate=args.num_generate,
        sequence_length=args.seq_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        penalty_window=args.penalty_window,
        jitter_method=args.jitter_method,
        tempo=args.tempo,
        humanize_tempo=not args.no_humanize_tempo
    )
