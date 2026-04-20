# ============================================
# VOICE ANALYSIS — EXTENDED VERSION
# Timbre | Voice Quality | Pauses & Rhythm
# ============================================

# ============================================
# PART 1. IMPORTS AND CONFIG
# ============================================
import os
import json
import tempfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from matplotlib.patches import Patch

AUDIO_FILE = "sample.m4a"
TARGET_SR = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512
FMIN = 50
FMAX = 400

# Silence threshold: frames with RMS below this level are treated as pauses
SILENCE_THRESHOLD_DB = -40   # dB; lower (e.g. -35) if background noise is high

# ============================================
# PART 2. LOAD AUDIO
# ============================================
if not os.path.exists(AUDIO_FILE):
    raise FileNotFoundError(f"File not found: {AUDIO_FILE}")

y, sr = librosa.load(AUDIO_FILE, sr=TARGET_SR, mono=True)
duration_sec = librosa.get_duration(y=y, sr=sr)
print(f"Loaded: {AUDIO_FILE} | SR={sr} | Duration={duration_sec:.2f}s")

# Praat (parselmouth) does not support .m4a / .mp3 directly.
# We write a temporary WAV from the already-decoded librosa array —
# no extra ffmpeg call needed.
_wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
WAV_FILE = _wav_tmp.name
_wav_tmp.close()

sf.write(WAV_FILE, y, sr, subtype="PCM_16")
print(f"Temporary WAV written for Praat: {WAV_FILE}")

# Parselmouth Sound object — reads from the WAV file
snd = parselmouth.Sound(WAV_FILE)

# ============================================
# PART 3. SEGMENTATION (equal-length chunks)
# ============================================
N_PARTS = 5
samples_per_part = len(y) // N_PARTS
segments = []

for i in range(N_PARTS):
    start = i * samples_per_part
    end = (i + 1) * samples_per_part if i < N_PARTS - 1 else len(y)
    segments.append({
        "part": i + 1,
        "start_time": start / sr,
        "end_time": end / sr,
        "audio": y[start:end]
    })

# ============================================
# PART 4. ANALYSIS FUNCTIONS
# ============================================

# --- 4A. TIMBRE ---
# Extends baseline MFCCs with:
#   delta-MFCCs      — rate of change of timbre frame-to-frame
#   Spectral contrast — depth of spectral peaks vs valleys across bands
#   Chroma            — tonal energy distribution across pitch classes

def analyze_timbre(audio, sr, hop_length=512):
    """
    Timbre is the spectral 'colour' of a voice beyond pitch and loudness.
    Delta-MFCCs capture how quickly timbre changes over time.
    Spectral contrast measures the relief of the spectrum (peaks vs valleys).
    Chroma variance reflects how tonally focused the voice is.
    """
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Spectral contrast across 7 frequency bands
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)

    # Chroma STFT — 12 pitch classes
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)

    result = {}

    # MFCC summary statistics (mean + std per coefficient)
    for i in range(mfcc.shape[0]):
        result[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        result[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    # Delta-MFCC: mean absolute movement of timbre
    result["delta_mfcc_mean_abs"]  = float(np.mean(np.abs(delta_mfcc)))
    result["delta2_mfcc_mean_abs"] = float(np.mean(np.abs(delta2_mfcc)))

    # Spectral contrast: mean per band
    for b in range(contrast.shape[0]):
        result[f"spec_contrast_band{b+1}"] = float(np.mean(contrast[b]))

    # Chroma variance: how focused the voice is on particular pitches
    result["chroma_variance"] = float(np.var(np.mean(chroma, axis=1)))

    return result


# --- 4B. VOICE QUALITY ---
# Formants F1/F2/F3 via Praat — vocal tract resonances, characterise vowels
# HNR (Harmonics-to-Noise Ratio) — voice clarity / breathiness
# Jitter  — cycle-to-cycle F0 instability (roughness, tremor)
# Shimmer — cycle-to-cycle amplitude instability

def analyze_voice_quality(snd_full, start_time, end_time):
    """
    Uses Praat through parselmouth.
    Extracts a segment of the Sound object and computes voice biomarkers.
    """
    part_snd = snd_full.extract_part(
        from_time=start_time,
        to_time=end_time,
        preserve_times=False
    )

    result = {}

    # --- Formants ---
    try:
        formant  = call(part_snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        duration = end_time - start_time
        n_points = max(1, int(duration / 0.01))  # sample every 10 ms

        f1_vals, f2_vals, f3_vals = [], [], []
        for t_idx in range(n_points):
            t  = t_idx * 0.01
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            if not np.isnan(f1): f1_vals.append(f1)
            if not np.isnan(f2): f2_vals.append(f2)
            if not np.isnan(f3): f3_vals.append(f3)

        result["F1_mean_hz"] = float(np.mean(f1_vals)) if f1_vals else np.nan
        result["F2_mean_hz"] = float(np.mean(f2_vals)) if f2_vals else np.nan
        result["F3_mean_hz"] = float(np.mean(f3_vals)) if f3_vals else np.nan
        result["F1_std_hz"]  = float(np.std(f1_vals))  if f1_vals else np.nan
        result["F2_std_hz"]  = float(np.std(f2_vals))  if f2_vals else np.nan

    except Exception as e:
        print(f"  [Formant error] {e}")
        result.update({"F1_mean_hz": np.nan, "F2_mean_hz": np.nan,
                        "F3_mean_hz": np.nan, "F1_std_hz": np.nan, "F2_std_hz": np.nan})

    # --- HNR (Harmonics-to-Noise Ratio) ---
    try:
        harmonicity = call(part_snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        result["HNR_dB"] = float(hnr) if not np.isnan(hnr) else np.nan
    except Exception as e:
        print(f"  [HNR error] {e}")
        result["HNR_dB"] = np.nan

    # --- Jitter and Shimmer ---
    try:
        pitch         = call(part_snd, "To Pitch", 0, FMIN, FMAX)
        point_process = call([part_snd, pitch], "To PointProcess (cc)")

        # Local jitter: % cycle-to-cycle F0 variation
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        result["jitter_local_pct"] = float(jitter_local * 100) if jitter_local else np.nan

        # Local shimmer in dB: amplitude cycle-to-cycle variation
        shimmer_dB = call([part_snd, point_process],
                          "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        result["shimmer_local_dB"] = float(shimmer_dB) if shimmer_dB else np.nan

    except Exception as e:
        print(f"  [Jitter/Shimmer error] {e}")
        result["jitter_local_pct"] = np.nan
        result["shimmer_local_dB"] = np.nan

    return result


# --- 4C. PAUSES AND SPEECH RHYTHM ---
# Detects silence based on frame-level RMS energy.
# Returns per-segment statistics and a list of individual pause events.

def analyze_pauses(audio, sr, threshold_db=SILENCE_THRESHOLD_DB,
                   hop_length=512, min_pause_frames=3):
    """
    Detects pauses as contiguous regions where RMS energy < threshold_db.
    min_pause_frames: minimum consecutive silent frames to count as a pause
                      (filters out micro-silences between phonemes).
    """
    rms    = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    is_silent      = rms_db < threshold_db
    frame_duration = hop_length / sr

    pauses    = []
    in_pause  = False
    pause_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_pause:
            in_pause    = True
            pause_start = i
        elif not silent and in_pause:
            pause_len = i - pause_start
            if pause_len >= min_pause_frames:
                pauses.append({
                    "start_sec":    pause_start * frame_duration,
                    "end_sec":      i * frame_duration,
                    "duration_sec": pause_len * frame_duration
                })
            in_pause = False

    # Handle a pause that reaches the very end of the segment
    if in_pause:
        pause_len = len(is_silent) - pause_start
        if pause_len >= min_pause_frames:
            pauses.append({
                "start_sec":    pause_start * frame_duration,
                "end_sec":      len(is_silent) * frame_duration,
                "duration_sec": pause_len * frame_duration
            })

    total_dur   = len(audio) / sr
    total_pause = sum(p["duration_sec"] for p in pauses)
    speech_dur  = total_dur - total_pause

    stats = {
        "n_pauses":          len(pauses),
        "total_pause_sec":   float(total_pause),
        "speech_ratio":      float(speech_dur / total_dur) if total_dur > 0 else np.nan,
        "mean_pause_sec":    float(np.mean([p["duration_sec"] for p in pauses])) if pauses else 0.0,
        "max_pause_sec":     float(max([p["duration_sec"]  for p in pauses]))    if pauses else 0.0,
        "pauses_per_second": float(len(pauses) / speech_dur) if speech_dur > 0 else np.nan,
    }

    return stats, pauses


# ============================================
# PART 5. RUN ANALYSIS ACROSS ALL SEGMENTS
# ============================================
all_results       = []
all_pauses_detail = []

for s in segments:
    print(f"\n--- Part {s['part']} ({s['start_time']:.2f}s – {s['end_time']:.2f}s) ---")

    timbre             = analyze_timbre(s["audio"], sr, HOP_LENGTH)
    quality            = analyze_voice_quality(snd, s["start_time"], s["end_time"])
    pause_stats, pause_list = analyze_pauses(s["audio"], sr)

    row = {
        "part":           s["part"],
        "start_time_sec": s["start_time"],
        "end_time_sec":   s["end_time"],
        **timbre,
        **quality,
        **pause_stats
    }
    all_results.append(row)

    for p in pause_list:
        all_pauses_detail.append({
            "part":      s["part"],
            "abs_start": s["start_time"] + p["start_sec"],
            **p
        })

results_df = pd.DataFrame(all_results)
pauses_df  = pd.DataFrame(all_pauses_detail)

print("\n\n=== KEY RESULTS ===")
key_cols = ["part", "F1_mean_hz", "F2_mean_hz", "F3_mean_hz",
            "HNR_dB", "jitter_local_pct", "shimmer_local_dB",
            "n_pauses", "speech_ratio", "mean_pause_sec"]
print(results_df[key_cols].to_string(index=False))

# ============================================
# PART 6. SAVE RESULTS
# ============================================
results_df.to_csv("voice_analysis_extended.csv", index=False)
pauses_df.to_csv("voice_pauses_detail.csv", index=False)
print("\nSaved: voice_analysis_extended.csv, voice_pauses_detail.csv")

# ============================================
# PART 7. VISUALISATIONS — main dashboard
# ============================================
parts = results_df["part"]
fig = plt.figure(figsize=(16, 18))
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- 7A. Formants F1 / F2 / F3 over segments ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(parts, results_df["F1_mean_hz"], "o-", label="F1", color="#E24B4A")
ax1.plot(parts, results_df["F2_mean_hz"], "o-", label="F2", color="#378ADD")
ax1.plot(parts, results_df["F3_mean_hz"], "o-", label="F3", color="#1D9E75")
ax1.set_title("Formants F1 / F2 / F3 (vocal tract resonances)")
ax1.set_xlabel("Recording segment")
ax1.set_ylabel("Frequency [Hz]")
ax1.legend()
ax1.grid(alpha=0.3)

# --- 7B. Vowel space: F1 vs F2 ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(results_df["F2_mean_hz"], results_df["F1_mean_hz"],
            c=parts, cmap="plasma", s=120, zorder=3)
for _, row in results_df.iterrows():
    ax2.annotate(f"P{int(row['part'])}", (row["F2_mean_hz"], row["F1_mean_hz"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=9)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.set_xlabel("F2 [Hz] →")
ax2.set_ylabel("F1 [Hz] ↓")
ax2.set_title("Vowel space (F1 vs F2)\n(phonetic convention: both axes inverted)")
ax2.grid(alpha=0.3)

# --- 7C. HNR — voice clarity ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(parts, results_df["HNR_dB"], color="#7F77DD", edgecolor="white")
ax3.axhline(20, color="gray", linestyle="--", linewidth=1, label="typical healthy ~20 dB")
ax3.set_title("HNR — Harmonics-to-Noise Ratio\n(higher = cleaner, less breathy voice)")
ax3.set_xlabel("Recording segment")
ax3.set_ylabel("HNR [dB]")
ax3.legend()
ax3.grid(alpha=0.3, axis="y")

# --- 7D. Jitter and Shimmer ---
ax4   = fig.add_subplot(gs[1, 1])
x     = np.array(parts)
ax4.bar(x - 0.2, results_df["jitter_local_pct"], width=0.35,
        label="Jitter [%]", color="#E24B4A", alpha=0.85)
ax4_r = ax4.twinx()
ax4_r.bar(x + 0.2, results_df["shimmer_local_dB"], width=0.35,
           label="Shimmer [dB]", color="#EF9F27", alpha=0.85)
ax4.set_title("Jitter & Shimmer\n(cycle-to-cycle instability of F0 and amplitude)")
ax4.set_xlabel("Recording segment")
ax4.set_ylabel("Jitter [%]", color="#E24B4A")
ax4_r.set_ylabel("Shimmer [dB]", color="#EF9F27")
ax4.set_xticks(x)
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_r.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax4.grid(alpha=0.3, axis="y")

# --- 7E. Pause count per segment ---
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(parts, results_df["n_pauses"], color="#1D9E75", edgecolor="white")
ax5.set_title("Number of pauses per segment")
ax5.set_xlabel("Recording segment")
ax5.set_ylabel("Pause count")
ax5.grid(alpha=0.3, axis="y")

# --- 7F. Speech ratio ---
ax6 = fig.add_subplot(gs[2, 1])
ax6.bar(parts, results_df["speech_ratio"] * 100, color="#378ADD", edgecolor="white")
ax6.axhline(70, color="gray", linestyle="--", linewidth=1, label="70% speech benchmark")
ax6.set_ylim(0, 105)
ax6.set_title("Speech ratio [%]\n(proportion of segment that is voiced)")
ax6.set_xlabel("Recording segment")
ax6.set_ylabel("% speech time")
ax6.legend()
ax6.grid(alpha=0.3, axis="y")

# --- 7G. Full waveform timeline with pauses highlighted ---
ax7 = fig.add_subplot(gs[3, :])
times = np.linspace(0, duration_sec, len(y))
ax7.plot(times, y, color="#888780", alpha=0.5, linewidth=0.4)

for _, p in pauses_df.iterrows():
    ax7.axvspan(p["abs_start"], p["abs_start"] + p["duration_sec"],
                alpha=0.35, color="#E24B4A", label="_nolegend_")

for s in segments[1:]:
    ax7.axvline(s["start_time"], color="#7F77DD", linestyle="--", linewidth=1, alpha=0.7)

ax7.set_title("Full waveform with detected pauses (red) and segment boundaries (purple)")
ax7.set_xlabel("Time [s]")
ax7.set_ylabel("Amplitude")
ax7.set_xlim(0, duration_sec)
legend_elements = [
    Patch(facecolor="#E24B4A", alpha=0.35, label="Pause"),
    plt.Line2D([0], [0], color="#7F77DD", linestyle="--", label="Segment boundary")
]
ax7.legend(handles=legend_elements, loc="upper right")
ax7.grid(alpha=0.2)

plt.suptitle("Voice Analysis — Timbre | Voice Quality | Pauses", fontsize=14, y=1.01)
plt.savefig("voice_analysis_extended.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: voice_analysis_extended.png")

# ============================================
# PART 8. TIMBRE DETAIL — spectral contrast + delta-MFCC
# ============================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Spectral contrast heatmap (frequency bands × segments)
contrast_cols = [c for c in results_df.columns if c.startswith("spec_contrast_band")]
contrast_data = results_df[contrast_cols].values.T
im = axes[0].imshow(contrast_data, aspect="auto", cmap="coolwarm",
                     extent=[1, N_PARTS + 0.5, 0.5, len(contrast_cols) + 0.5])
axes[0].set_xlabel("Recording segment")
axes[0].set_ylabel("Frequency band")
axes[0].set_title("Spectral contrast per band\n(timbre — spectral relief)")
plt.colorbar(im, ax=axes[0], label="Contrast [dB]")

# Delta and Delta² MFCC — timbre dynamics
axes[1].bar(parts, results_df["delta_mfcc_mean_abs"],
            label="Delta-MFCC", color="#7F77DD", alpha=0.85)
axes[1].bar(parts, results_df["delta2_mfcc_mean_abs"],
            bottom=results_df["delta_mfcc_mean_abs"],
            label="Delta²-MFCC", color="#D4537E", alpha=0.85)
axes[1].set_title("Timbre dynamics (Delta and Delta² MFCC)\n(higher = faster timbre change)")
axes[1].set_xlabel("Recording segment")
axes[1].set_ylabel("Mean absolute value")
axes[1].legend()
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("voice_timbre_detail.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: voice_timbre_detail.png")

# ============================================
# PART 9. CLEANUP TEMPORARY WAV
# ============================================
os.remove(WAV_FILE)
print(f"Temporary WAV removed: {WAV_FILE}")
