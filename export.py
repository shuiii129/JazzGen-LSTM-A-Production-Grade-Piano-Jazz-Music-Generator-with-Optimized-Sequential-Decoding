import random
import music21

def token_to_element(token, velocity_range=(60, 100)):
    """
    Decodes a token string back into a music21 Note, Chord, or Rest.
    Applies a dynamic velocity within velocity_range for notes and chords to humanize the performance.
    """
    parts = token.split('_')
    if len(parts) < 2:
        return None
        
    event_type = parts[0]
    duration_val = float(parts[-1])
    
    if event_type == "Note":
        pitch_val = int(parts[1])
        n = music21.note.Note(pitch_val)
        n.duration.quarterLength = duration_val
        # Assign variable velocity to humanize notes
        n.volume.velocity = random.randint(*velocity_range)
        return n
        
    elif event_type == "Chord":
        pitches_str = parts[1]
        pitches = [int(p) for p in pitches_str.split('.')]
        c = music21.chord.Chord(pitches)
        c.duration.quarterLength = duration_val
        
        # Assign slightly varying velocities to individual notes in the chord for organic feel
        base_velocity = random.randint(*velocity_range)
        for n in c.notes:
            n.volume.velocity = max(1, min(127, base_velocity + random.randint(-5, 5)))
        return c
        
    elif event_type == "Rest":
        r = music21.note.Rest()
        r.duration.quarterLength = duration_val
        return r
        
    return None

def tokens_to_stream(tokens, tempo=120, velocity_range=(60, 100), humanize_tempo=True):
    """
    Decodes a token sequence into a music21.stream.Stream.
    Specifies initial tempo, optionally adds expressive tempo rubato, and parses notes.
    """
    stream = music21.stream.Stream()
    
    # Set initial tempo
    tm = music21.tempo.MetronomeMark(number=tempo)
    stream.append(tm)
    
    current_tempo = tempo
    notes_since_tempo_change = 0
    
    for token in tokens:
        if token in ["<PAD>", "<UNK>"]:
            continue
            
        element = token_to_element(token, velocity_range)
        if element is not None:
            stream.append(element)
            
            # Optionally add subtle tempo changes (rubato) to jazz up the performance
            if humanize_tempo and isinstance(element, (music21.note.Note, music21.chord.Chord)):
                notes_since_tempo_change += 1
                # Every ~15 events, introduce a minor tempo fluctuation
                if notes_since_tempo_change >= 15:
                    tempo_delta = random.randint(-4, 4)
                    current_tempo = max(60, min(200, tempo + tempo_delta))
                    stream.append(music21.tempo.MetronomeMark(number=current_tempo))
                    notes_since_tempo_change = 0
                    
    return stream

def export_tokens_to_midi(tokens, output_path, tempo=120, velocity_range=(60, 100), humanize_tempo=True):
    """
    Main export entry point to serialize a generated token stream to a MIDI file.
    """
    stream = tokens_to_stream(tokens, tempo, velocity_range, humanize_tempo)
    stream.write('midi', fp=output_path)
    print(f"Successfully exported {len(tokens)} tokens to MIDI: {output_path}")
