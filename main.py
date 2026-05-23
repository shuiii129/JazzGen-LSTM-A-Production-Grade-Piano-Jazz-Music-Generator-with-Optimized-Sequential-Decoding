import sys
import argparse
from train import train
from evaluate import evaluate
from generate import generate

def main():
    parser = argparse.ArgumentParser(
        description="Attention-LSTM Jazz Music Generator CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")
    
    # 1. Subparser for train
    train_parser = subparsers.add_parser("train", help="Train the Attention-LSTM music model")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing MIDI files")
    train_parser.add_argument("--model_dir", type=str, default="checkpoints", help="Directory to save checkpoint, vocab, and metrics")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--seq_len", type=int, default=100, help="Sequence length")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding size")
    train_parser.add_argument("--hidden_dim", type=int, default=256, help="LSTM hidden units")
    train_parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked LSTM layers")
    train_parser.add_argument("--attention_dim", type=int, default=128, help="Self-Attention dimension")
    train_parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")
    train_parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split fraction")
    train_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience epochs")
    train_parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")

    # 2. Subparser for evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained Attention-LSTM performance")
    eval_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    eval_parser.add_argument("--vocab", type=str, default="checkpoints/vocab.pkl", help="Path to vocabulary dictionary")
    eval_parser.add_argument("--data_dir", type=str, required=True, help="Path to evaluation MIDI files folder")
    eval_parser.add_argument("--seq_len", type=int, default=100, help="Sequence length of model inputs")

    # 3. Subparser for generate
    gen_parser = subparsers.add_parser("generate", help="Generate MIDI music using the trained model")
    gen_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    gen_parser.add_argument("--vocab", type=str, default="checkpoints/vocab.pkl", help="Path to vocabulary dictionary")
    gen_parser.add_argument("--output", type=str, default="output.mid", help="Path to save output MIDI file")
    gen_parser.add_argument("--seed_midi", type=str, default=None, help="Path to seed MIDI file (optional)")
    gen_parser.add_argument("--num_generate", type=int, default=200, help="Number of tokens to generate")
    gen_parser.add_argument("--seq_len", type=int, default=100, help="Sequence length model uses")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling")
    gen_parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering candidate pool size (default 0, disabled when top_p is active)")
    gen_parser.add_argument("--top_p", type=float, default=0.90, help="Top-p (nucleus) filtering threshold (default 0.90)")
    gen_parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty factor (default 1.2)")
    gen_parser.add_argument("--penalty_window", type=int, default=10, help="Look back window for repetition penalty (default 10)")
    gen_parser.add_argument("--jitter_method", type=str, choices=["second", "random", "none"], default="second", help="Method to escape stuck loops (default 'second')")
    gen_parser.add_argument("--tempo", type=int, default=120, help="Default tempo (BPM)")
    gen_parser.add_argument("--no_humanize_tempo", action="store_true", help="Disable expressive tempo changes")
    
    args = parser.parse_args()
    
    if args.command == "train":
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
    elif args.command == "evaluate":
        evaluate(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            data_dir=args.data_dir,
            seq_len=args.seq_len
        )
    elif args.command == "generate":
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

if __name__ == "__main__":
    main()
