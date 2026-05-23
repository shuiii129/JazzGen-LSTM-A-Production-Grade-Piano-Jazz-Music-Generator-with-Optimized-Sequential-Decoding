import os
import argparse
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from scipy.io import wavfile

def generate_piano_roll_plot(pm, midi_path, output_plot_path):
    """
    Plots the MIDI notes as a visual piano roll (x-axis: time, y-axis: pitch).
    Uses color variations for different instruments or note velocities.
    """
    plt.figure(figsize=(14, 6))
    
    # Track note counts
    note_count = 0
    
    for instrument in pm.instruments:
        name = instrument.name if instrument.name else "Piano"
        if instrument.is_drum:
            continue
            
        notes = instrument.notes
        note_count += len(notes)
        
        # Gather note coordinates
        for note in notes:
            # Draw note duration line segment
            plt.plot(
                [note.start, note.end], 
                [note.pitch, note.pitch], 
                color="#6200ea" if not instrument.is_drum else "#e91e63", 
                linewidth=4.5, 
                solid_capstyle="round",
                alpha=0.85
            )
            # Add a small scatter marker at the onset for visual accent
            plt.scatter(
                note.start, 
                note.pitch, 
                color="#03dac6", 
                s=20, 
                zorder=3
            )
            
    # Style the plot for a sleek, dark/modern look
    plt.title(f"Piano Roll Visualizer — {os.path.basename(midi_path)}", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Time (seconds)", fontsize=12, labelpad=10)
    plt.ylabel("MIDI Pitch", fontsize=12, labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.5, color="gray")
    
    # Set y-limits slightly padded around note pitches
    all_pitches = [n.pitch for inst in pm.instruments for n in inst.notes if not inst.is_drum]
    if all_pitches:
        plt.ylim(min(all_pitches) - 2, max(all_pitches) + 2)
        
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    print(f"Successfully saved piano roll plot ({note_count} notes) to {output_plot_path}")
    plt.close()

def synthesize_midi_to_wav(pm, output_wav_path, sample_rate=44100):
    """
    Synthesizes MIDI notes to a raw audio waveform using pretty_midi's native sine synthesis.
    Requires no external binaries (like fluidsynth) or SoundFont paths, making it highly portable.
    """
    print("Synthesizing MIDI to audio waveform (native sine-wave synthesis)...")
    
    # Synthesize to raw audio signal
    audio_data = pm.synthesize(fs=sample_rate)
    
    # Normalize audio to prevent clipping and fit the int16 range
    if len(audio_data) > 0:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
    # Convert to 16-bit PCM integer WAV format
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Write WAV file
    wavfile.write(output_wav_path, sample_rate, audio_int16)
    print(f"Successfully synthesized and saved WAV audio file to {output_wav_path}")

def main():
    parser = argparse.ArgumentParser(
        description="MIDI Music Visualizer & Audio Synthesizer Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--midi", type=str, required=True, help="Path to input MIDI file (.mid/.midi)")
    parser.add_argument("--plot", type=str, default=None, help="Path to save rendered piano roll plot PNG image (optional)")
    parser.add_argument("--wav", type=str, default=None, help="Path to save synthesized WAV audio file (optional)")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Synthesized audio sample rate in Hz")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.midi):
        raise FileNotFoundError(f"MIDI file not found at: {args.midi}")
        
    # Load PrettyMIDI file
    print(f"Loading MIDI file: {args.midi}")
    try:
        pm = pretty_midi.PrettyMIDI(args.midi)
    except Exception as e:
        print(f"Error parsing MIDI file: {e}")
        return
        
    # Process actions
    if args.plot:
        generate_piano_roll_plot(pm, args.midi, args.plot)
        
    if args.wav:
        synthesize_midi_to_wav(pm, args.wav, sample_rate=args.sample_rate)
        
    if not args.plot and not args.wav:
        print("Note: Specify either --plot [file.png] or --wav [file.wav] to output visual/audio files.")

if __name__ == "__main__":
    main()
