import os
import argparse
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import prepare_data
from model import AttentionLSTM

def train(data_dir, model_dir, epochs=20, batch_size=64, seq_len=100, lr=0.001, embedding_dim=128, hidden_dim=256, num_layers=2, attention_dim=128, dropout=0.4, validation_split=0.2, patience=10, weight_decay=1e-4):
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Prepare data (parsing MIDI files and creating DataLoaders)
    train_loader, val_loader, token_to_idx, idx_to_token = prepare_data(data_dir, sequence_length=seq_len, batch_size=batch_size, validation_split=validation_split)
    
    # Save vocabulary dictionary for inference
    vocab_path = os.path.join(model_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({"token_to_idx": token_to_idx, "idx_to_token": idx_to_token}, f)
    print(f"Successfully saved vocabulary to {vocab_path}")
    
    vocab_size = len(token_to_idx)
    
    # 2. Setup Device & Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AttentionLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        attention_dim=attention_dim,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 3. Training loop
    print("Starting training loop...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Track metrics history over epochs for plotting
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_loss = 0.0
        batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Flatten sequence predictions and targets for CrossEntropyLoss
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
            # Calculate next-token accuracy ignoring padding token (0)
            preds = torch.argmax(logits, dim=-1)
            mask = (targets != 0)
            correct_predictions += ((preds == targets) & mask).sum().item()
            total_predictions += mask.sum().item()
            
        if batches == 0:
            print("No training batches created. Please verify data.")
            break
            
        avg_loss = total_loss / batches
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                val_logits = model(val_inputs)
                v_loss = criterion(val_logits.view(-1, vocab_size), val_targets.view(-1))
                
                val_loss += v_loss.item()
                val_batches += 1
                
                val_preds = torch.argmax(val_logits, dim=-1)
                val_mask = (val_targets != 0)
                val_correct += ((val_preds == val_targets) & val_mask).sum().item()
                val_total += val_mask.sum().item()
                
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 100 else float('inf')
        
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {avg_loss:.4f} | Train Acc: {accuracy*100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}% | "
            f"Val Perp: {val_perplexity:.2f}"
        )
        
        # Append to metrics history
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(accuracy)
        history["val_loss"].append(avg_val_loss if val_batches > 0 else None)
        history["val_acc"].append(val_accuracy if val_batches > 0 else None)
        
        # Save metrics to json file
        import json
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w") as json_file:
            json.dump(history, json_file, indent=4)
            
        # Checkpoint if we achieved a better validation loss (or training loss if validation is not used)
        track_metric = avg_val_loss if val_batches > 0 else avg_loss
        if track_metric < best_val_loss:
            best_val_loss = track_metric
            epochs_no_improve = 0
            checkpoint_path = os.path.join(model_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'accuracy': val_accuracy,
                'perplexity': val_perplexity,
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'attention_dim': attention_dim,
                'dropout': dropout,
                'seq_len': seq_len
            }, checkpoint_path)
            print(f" --> Saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                metric_name = "Validation Loss" if val_batches > 0 else "Train Loss"
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement in {metric_name} for {patience} epochs).")
                break
                
    print("Training process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch Attention-LSTM Music Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing MIDI files")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Directory to save checkpoint and vocabulary")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden units")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked LSTM layers")
    parser.add_argument("--attention_dim", type=int, default=128, help="Self-Attention dimension")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        validation_split=args.validation_split,
        patience=args.patience,
        weight_decay=args.weight_decay
    )
