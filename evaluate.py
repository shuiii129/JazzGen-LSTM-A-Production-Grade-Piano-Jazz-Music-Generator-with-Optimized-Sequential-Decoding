import os
import argparse
import pickle
import torch
from data_pipeline import prepare_data
from model import AttentionLSTM

def evaluate(checkpoint_path, vocab_path, data_dir, seq_len=100):
    # 1. Load Vocab Mappings
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}.")
        
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
    token_to_idx = vocab_data["token_to_idx"]
    idx_to_token = {int(k): v for k, v in vocab_data["idx_to_token"].items()}
    vocab_size = len(token_to_idx)
    
    # 2. Load Model Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = AttentionLSTM(
        vocab_size=vocab_size,
        embedding_dim=checkpoint.get('embedding_dim', 128),
        hidden_dim=checkpoint.get('hidden_dim', 256),
        num_layers=checkpoint.get('num_layers', 2),
        attention_dim=checkpoint.get('attention_dim', 128),
        dropout=checkpoint.get('dropout', 0.3)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Load Evaluation Data
    # Use batch_size = 1 for simple tracking
    train_loader, _, _, _ = prepare_data(data_dir, sequence_length=seq_len, batch_size=1, validation_split=0.0, token_to_idx=token_to_idx)
    
    all_targets = []
    all_predictions = []
    
    print("Running evaluation inference...")
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            logits = model(inputs)  # Shape: [1, seq_len, vocab_size]
            
            # Flatten along batch and sequence length
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            
            preds = torch.argmax(logits_flat, dim=-1).cpu().tolist()
            targets_list = targets_flat.cpu().tolist()
            
            # Filter out padding tokens (index 0)
            for t, p in zip(targets_list, preds):
                if t != 0:
                    all_targets.append(t)
                    all_predictions.append(p)
            
    if not all_targets:
        print("No validation samples evaluated.")
        return
        
    # 4. Compute Metrics
    total_samples = len(all_targets)
    correct = sum(1 for t, p in zip(all_targets, all_predictions) if t == p)
    overall_accuracy = correct / total_samples
    
    print("\n" + "=" * 90)
    print(f"EVALUATION REPORT FOR CHECKPOINT: {checkpoint_path}")
    print(f"Overall Dataset Accuracy: {overall_accuracy*100:.2f}% (Total Samples: {total_samples})")
    print("=" * 90 + "\n")
    
    # Gather all unique indices from both targets and predictions
    unique_indices = sorted(list(set(all_targets + all_predictions)))
    
    # Table Header
    print(f"{'Token Name':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8} | {'Distribution %':<14}")
    print("-" * 92)
    
    for idx in unique_indices:
        token_name = idx_to_token.get(idx, f"Idx_{idx}")
        
        # Calculate True Positives, False Positives, False Negatives
        tp = sum(1 for t, p in zip(all_targets, all_predictions) if t == idx and p == idx)
        fp = sum(1 for t, p in zip(all_targets, all_predictions) if t != idx and p == idx)
        fn = sum(1 for t, p in zip(all_targets, all_predictions) if t == idx and p != idx)
        support = sum(1 for t in all_targets if t == idx)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        dist_pct = (support / total_samples) * 100
        
        print(f"{token_name:<28} | {precision:<10.2f} | {recall:<10.2f} | {f1:<10.2f} | {support:<8} | {dist_pct:<13.2f}%")
        
    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Attention-LSTM performance on a dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to best_model.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.pkl", help="Path to vocab.pkl")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to evaluation MIDI files folder")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length of model inputs")
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        data_dir=args.data_dir,
        seq_len=args.seq_len
    )
