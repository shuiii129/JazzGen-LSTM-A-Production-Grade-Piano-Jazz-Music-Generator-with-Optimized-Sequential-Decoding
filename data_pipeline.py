import os
import pickle
import music21
import torch
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def parse_midi_to_tokens(file_path):
    """
    Parses a MIDI file and extracts Note, Chord, and Rest events into a sequence of token strings.
    Uses chordify to handle overlapping notes and resolve polyphony.
    """
    try:
        score = music21.converter.parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []
    
    try:
        # chordify collapses multiple parts into a single part with chords and rests
        chordified = score.chordify()
        elements = chordified.flatten().notesAndRests
    except Exception as e:
        print(f"Error chordifying {file_path}, falling back to flat stream: {e}")
        elements = score.flatten().notesAndRests
        
    tokens = []
    for element in elements:
        duration = round(float(element.duration.quarterLength), 4)
        if isinstance(element, music21.chord.Chord):
            pitches = sorted([p.midi for p in element.pitches])
            if len(pitches) == 0:
                continue
            elif len(pitches) == 1:
                tokens.append(f"Note_{pitches[0]}_{duration}")
            else:
                pitches_str = ".".join(map(str, pitches))
                tokens.append(f"Chord_{pitches_str}_{duration}")
        elif isinstance(element, music21.note.Note):
            pitch = element.pitch.midi
            tokens.append(f"Note_{pitch}_{duration}")
        elif isinstance(element, music21.note.Rest):
            tokens.append(f"Rest_{duration}")
            
    return tokens

def expand_vocab_with_augmentations(all_tokens, common_durations):
    expanded_tokens = set(all_tokens)
    
    # Also collect all durations in the original tokens
    original_durations = set()
    for token in all_tokens:
        parts = token.split('_')
        if len(parts) >= 2:
            try:
                original_durations.add(float(parts[-1]))
            except ValueError:
                pass
                
    all_possible_durations = list(set(common_durations) | original_durations)
    
    for token in all_tokens:
        if token in [PAD_TOKEN, UNK_TOKEN]:
            continue
        parts = token.split('_')
        if len(parts) < 2:
            continue
        event_type = parts[0]
        
        if event_type == "Rest":
            for d in all_possible_durations:
                expanded_tokens.add(f"Rest_{d}")
        elif event_type == "Note":
            try:
                pitch = int(parts[1])
                for s in range(-5, 7): # -5 to +6
                    p_trans = max(0, min(127, pitch + s))
                    for d in all_possible_durations:
                        expanded_tokens.add(f"Note_{p_trans}_{d}")
            except ValueError:
                pass
        elif event_type == "Chord":
            try:
                pitches_str = parts[1]
                pitches = [int(p) for p in pitches_str.split('.')]
                for s in range(-5, 7): # -5 to +6
                    p_trans_list = sorted([max(0, min(127, p + s)) for p in pitches])
                    p_trans_str = ".".join(map(str, p_trans_list))
                    for d in all_possible_durations:
                        expanded_tokens.add(f"Chord_{p_trans_str}_{d}")
            except ValueError:
                pass
                
    return sorted(list(expanded_tokens))

def build_vocab(all_tokens):
    """
    Builds a unified integer-token dictionary from all extracted tokens,
    including expanded tokens for augmentation.
    """
    common_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    expanded_tokens = expand_vocab_with_augmentations(all_tokens, common_durations)
    
    token_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, token in enumerate(expanded_tokens, start=2):
        token_to_idx[token] = idx
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token

class MidiDataset(Dataset):
    def __init__(self, token_sequences, token_to_idx, sequence_length=100, augment=False):
        self.token_to_idx = token_to_idx
        self.sequence_length = sequence_length
        self.augment = augment
        self.inputs = []
        self.targets = []
        
        # Extract possible durations from vocab for jittering
        self.possible_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        for token in token_to_idx.keys():
            parts = token.split('_')
            if len(parts) >= 2:
                try:
                    self.possible_durations.append(float(parts[-1]))
                except ValueError:
                    pass
        self.possible_durations = sorted(list(set(self.possible_durations)))
        
        for seq in token_sequences:
            if len(seq) < 2:
                continue
            
            # If the sequence is shorter than sequence_length + 1, pad it
            if len(seq) < sequence_length + 1:
                padding_needed = (sequence_length + 1) - len(seq)
                padded_seq = [PAD_TOKEN] * padding_needed + seq
                self.inputs.append(padded_seq[:-1])
                self.targets.append(padded_seq[1:])
            else:
                # Slide over sequence
                for i in range(len(seq) - sequence_length):
                    self.inputs.append(seq[i : i + sequence_length])
                    self.targets.append(seq[i + 1 : i + sequence_length + 1])
                    
    def __len__(self):
        return len(self.inputs)
        
    def _augment_token(self, token, semitones, jitter_prob=0.3):
        if token in [PAD_TOKEN, UNK_TOKEN]:
            return token
        parts = token.split('_')
        if len(parts) < 2:
            return token
            
        event_type = parts[0]
        duration_val = float(parts[-1])
        
        # Jitter duration
        import random
        if random.random() < jitter_prob:
            duration_val = random.choice(self.possible_durations)
            
        if event_type == "Rest":
            return f"Rest_{duration_val}"
            
        elif event_type == "Note":
            try:
                pitch = int(parts[1])
                pitch_trans = max(0, min(127, pitch + semitones))
                return f"Note_{pitch_trans}_{duration_val}"
            except ValueError:
                return token
                
        elif event_type == "Chord":
            try:
                pitches = [int(p) for p in parts[1].split('.')]
                pitches_trans = sorted([max(0, min(127, p + semitones)) for p in pitches])
                pitches_str = ".".join(map(str, pitches_trans))
                return f"Chord_{pitches_str}_{duration_val}"
            except ValueError:
                return token
                
        return token
        
    def __getitem__(self, idx):
        seq = self.inputs[idx]
        target = self.targets[idx]
        
        if self.augment:
            import random
            semitones = random.randint(-5, 6)
            seq = [self._augment_token(t, semitones) for t in seq]
            target = [self._augment_token(t, semitones) for t in target]
            
        unk_idx = self.token_to_idx[UNK_TOKEN]
        input_indices = [self.token_to_idx.get(t, unk_idx) for t in seq]
        target_indices = [self.token_to_idx.get(t, unk_idx) for t in target]
        
        return (
            torch.tensor(input_indices, dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long)
        )

def prepare_data(data_dir, sequence_length=100, batch_size=64, validation_split=0.2, token_to_idx=None):
    """
    Loads all MIDI files from data_dir, parses them, and returns train/validation DataLoaders.
    If token_to_idx is provided, it uses it instead of rebuilding vocabulary.
    """
    midi_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.mid', '.midi'))]
    if not midi_files:
        raise ValueError(f"No MIDI files found in {data_dir}")
        
    print(f"Found {len(midi_files)} MIDI files. Parsing...")
    all_sequences = []
    all_tokens = []
    
    for f in midi_files:
        tokens = parse_midi_to_tokens(f)
        if tokens:
            all_sequences.append(tokens)
            all_tokens.extend(tokens)
            
    if token_to_idx is None:
        print(f"Extracted {len(all_tokens)} tokens in total. Building vocab...")
        token_to_idx, idx_to_token = build_vocab(all_tokens)
    else:
        print("Using pre-loaded vocabulary dictionary.")
        idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        
    print(f"Vocab size: {len(token_to_idx)}")
    
    # Train/Validation Split
    if len(all_sequences) > 1:
        split_idx = int(len(all_sequences) * (1 - validation_split))
        train_sequences = all_sequences[:split_idx]
        val_sequences = all_sequences[split_idx:]
    elif len(all_sequences) == 1:
        single_seq = all_sequences[0]
        split_idx = int(len(single_seq) * (1 - validation_split))
        train_sequences = [single_seq[:split_idx]]
        val_sequences = [single_seq[split_idx:]]
    else:
        train_sequences, val_sequences = [], []
        
    train_dataset = MidiDataset(train_sequences, token_to_idx, sequence_length, augment=(validation_split > 0))
    val_dataset = MidiDataset(val_sequences, token_to_idx, sequence_length, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(validation_split > 0), drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, token_to_idx, idx_to_token
