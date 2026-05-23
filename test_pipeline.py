import os
import shutil
import pickle
import torch
import music21
from data_pipeline import prepare_data, parse_midi_to_tokens
from model import AttentionLSTM
from train import train
from generate import generate

def create_synthetic_midi(file_path):
    """
    Creates a synthetic MIDI file with notes, chords, and rests for testing.
    """
    stream = music21.stream.Stream()
    
    # Add initial tempo
    stream.append(music21.tempo.MetronomeMark(number=120))
    
    # 1. Notes
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for pitch in pitches:
        n = music21.note.Note(pitch)
        n.duration.quarterLength = 1.0
        stream.append(n)
        
    # 2. Chord
    c = music21.chord.Chord([60, 64, 67])
    c.duration.quarterLength = 2.0
    stream.append(c)
    
    # 3. Rest
    r = music21.note.Rest()
    r.duration.quarterLength = 1.0
    stream.append(r)
    
    # 4. Another chord
    c2 = music21.chord.Chord([62, 66, 69])
    c2.duration.quarterLength = 1.5
    stream.append(c2)
    
    # Write to file
    stream.write('midi', fp=file_path)
    print(f"Created synthetic MIDI file at {file_path}")

def run_tests():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "test_data")
    model_dir = os.path.join(project_dir, "test_checkpoints")
    output_midi = os.path.join(project_dir, "test_output.mid")
    
    # Clean up previous test directories/files
    for d in [data_dir, model_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    if os.path.exists(output_midi):
        os.remove(output_midi)
        
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Generate synthetic MIDI files for training
    create_synthetic_midi(os.path.join(data_dir, "syn1.mid"))
    create_synthetic_midi(os.path.join(data_dir, "syn2.mid"))
    
    print("\n--- Testing Data Pipeline ---")
    # Parse tokens check
    tokens = parse_midi_to_tokens(os.path.join(data_dir, "syn1.mid"))
    print("Parsed tokens from syn1.mid:", tokens)
    assert len(tokens) > 0, "No tokens parsed!"
    assert any(t.startswith("Note") for t in tokens), "No Note tokens parsed!"
    assert any(t.startswith("Chord") for t in tokens), "No Chord tokens parsed!"
    assert any(t.startswith("Rest") for t in tokens), "No Rest tokens parsed!"
    print("Data parsing test passed!")
    
    # DataLoader preparation check
    # Set seq_len to 5 for fast testing since synthetic files are short
    seq_len = 5
    train_loader, val_loader, token_to_idx, idx_to_token = prepare_data(data_dir, sequence_length=seq_len, batch_size=4)
    print(f"Vocab size: {len(token_to_idx)}")
    inputs, targets = next(iter(train_loader))
    print(f"DataLoader batch shape: inputs={inputs.shape}, targets={targets.shape}")
    assert inputs.shape == (4, seq_len), f"Expected shape (4, {seq_len}), got {inputs.shape}"
    assert targets.shape == (4, seq_len), f"Expected shape (4, {seq_len}), got {targets.shape}"
    print("DataLoader shape test passed!")
    
    print("\n--- Testing Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionLSTM(
        vocab_size=len(token_to_idx),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        attention_dim=16,
        dropout=0.1
    ).to(device)
    
    inputs_device = inputs.to(device)
    logits = model(inputs_device)
    print("Model logits shape:", logits.shape)
    assert logits.shape == (4, seq_len, len(token_to_idx)), f"Expected shape (4, {seq_len}, {len(token_to_idx)}), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model logits contain NaN values!"
    print("Model forward pass test passed!")
    
    print("\n--- Testing Training Loop (1 epoch) ---")
    train(
        data_dir=data_dir,
        model_dir=model_dir,
        epochs=1,
        batch_size=4,
        seq_len=seq_len,
        lr=0.01,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        attention_dim=16,
        dropout=0.1
    )
    assert os.path.exists(os.path.join(model_dir, "vocab.pkl")), "vocab.pkl was not saved!"
    assert os.path.exists(os.path.join(model_dir, "best_model.pt")), "best_model.pt was not saved!"
    print("Training loop test passed!")
    
    print("\n--- Testing Generation & Export ---")
    generate(
        checkpoint_path=os.path.join(model_dir, "best_model.pt"),
        vocab_path=os.path.join(model_dir, "vocab.pkl"),
        output_path=output_midi,
        seed_midi=None,
        num_generate=20,
        sequence_length=seq_len,
        temperature=1.2,
        top_k=3,
        top_p=0.9,
        tempo=100,
        humanize_tempo=True
    )
    
    assert os.path.exists(output_midi), "Output MIDI file was not created!"
    assert os.path.getsize(output_midi) > 0, "Output MIDI file is empty!"
    print(f"Generation and export test passed! MIDI size: {os.path.getsize(output_midi)} bytes")
    
    print("\n==============================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==============================================")

if __name__ == "__main__":
    run_tests()
