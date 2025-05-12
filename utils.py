import numpy as np
import torch
import librosa
import os
from tqdm import tqdm
import json
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import librosa.display
import torch.nn.functional as F




SAMPLE_RATE = 16000
N_BINS = 256
CHUNK_SIZE = 16000  # 1 second

def quantize_audio(audio, n_bins=N_BINS):
    audio = np.clip(audio, -1.0, 1.0)
    audio_norm = (audio + 1.0) / 2.0
    return (audio_norm * (n_bins - 1)).astype(np.uint8)

def dequantize_audio(tokens, n_bins=N_BINS):
    audio_norm = tokens.astype(np.float32) / (n_bins - 1)
    return 2.0 * audio_norm - 1.0

def tokenize_directory(audio_dir):
    token_seqs = []
    for file in tqdm(os.listdir(audio_dir)):
        if file.endswith('.wav'):
            path = os.path.join(audio_dir, file)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            if len(audio) < CHUNK_SIZE:
                continue
            for i in range(0, len(audio) - CHUNK_SIZE + 1, CHUNK_SIZE):
                chunk = audio[i:i+CHUNK_SIZE]
                tokens = mu_law_encode(chunk) #originally quantize
                token_seqs.append(tokens)
    return token_seqs

@torch.no_grad()
def generate_conditional(model_path, label_id=0, seed_token=128, length=16000, temperature=1.0):
    """
    generates a sound from conditioned model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConditionalWaveNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tokens = [seed_token]
    label_tensor = torch.tensor([label_id], device=device)

    for _ in range(length - 1):
        x = torch.tensor(tokens[-1024:], dtype=torch.long, device=device).unsqueeze(0)
        x_input = F.one_hot(x, num_classes=256).float().permute(0, 2, 1)
        logits = model(x_input, label_tensor)
        probs = F.softmax(logits[:, :, -1] / temperature, dim=1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)

    waveform = mu_law_decode(np.array(tokens, dtype=np.uint8)) #replaced quantize
    return waveform
    
# def load_nsynth_labels(json_path, audio_dir):
#     with open(json_path, "r") as f:
#         metadata = json.load(f)

#     file_to_label = {}
#     for key, info in metadata.items():
#         filename = f"{key}.wav"
#         if os.path.exists(os.path.join(audio_dir, filename)):
#             file_to_label[filename] = info["instrument_family"]  # 0–10

#     return file_to_label  # dict: filename → label
    
# def tokenize_nsynth_with_labels(audio_dir, file_to_label):
#     token_seqs = []
#     labels = []

#     for file in tqdm(file_list):
#         path = os.path.join(audio_dir, file)
#         audio = load_and_trim_audio(path)
#         if len(audio) < chunk_size:
#             continue
#         for i in range(0, len(audio) - 16000 + 1, 16000):
#             chunk = audio[i:i+16000]
#             tokens = quantize(chunk)
#             token_seqs.append(torch.tensor(tokens, dtype=torch.long))
#             labels.append(file_to_label[file])

#     return token_seqs, labels

# def load_nsynth_labels(json_path, audio_dir):
#     with open(json_path, "r") as f:
#         metadata = json.load(f) 

#     file_to_label = {}
#     for key, info in metadata.items():
#         filename = key + ".wav"
#         if os.path.exists(os.path.join(audio_dir, filename)):
#             instrument_family = info["instrument_family"]
#             pitch = info["pitch"]
#             file_to_label[filename] = (instrument_family, pitch)
#     return file_to_label

# def tokenize_nsynth_with_labels(audio_dir, file_to_label):
#     token_seqs = []
#     labels = []
    
#     for file, (inst, pitch) in tqdm(file_to_label.items()):
#         path = os.path.join(audio_dir, file)
#         audio, sr = librosa.load(path, sr=16000, mono=True)
#         if len(audio) < 16000:
#             continue
#         for i in range(0, len(audio) - 16000 + 1, 16000):
#             chunk = audio[i:i+16000]
#             tokens = quantize(chunk)
#             token_seqs.append(torch.tensor(tokens, dtype=torch.long))
#             labels.append((inst, pitch)) 
#     return token_seqs, labels
    
@torch.no_grad()
def generate_conditioned(model, inst_id, pitch_id, length=16000, seed=128, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), temp=1.0):
    model.eval()
    x = torch.full((1, 1), seed, dtype=torch.long, device=device)  # (1, 1)
    inst = torch.tensor([inst_id], device=device)
    pitch = torch.tensor([pitch_id], device=device)

    samples = []

    for _ in range(length):
        x_input = F.one_hot(x[:, -1024:], num_classes=256).float().permute(0, 2, 1)  # (1, 256, T)
        logits = model(x_input, inst, pitch)
        probs = F.softmax(logits[:, :, -1] / temp, dim=-1)
        sample = torch.multinomial(probs, num_samples=1)
        samples.append(sample.item())
        x = torch.cat([x, sample], dim=1)

    tokens = np.array(samples, dtype=np.uint8)
    return mu_law_decode(tokens) # yl what i did here


def show_sample(model, inst_id, pitch_id, title=None):
    audio = generate_conditioned(model, inst_id, pitch_id, length=16000, device=device)
    
    display(Audio(audio, rate=16000))
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(D, sr=16000, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title or f"Inst {inst_id} - Pitch {pitch_id}")
    plt.tight_layout()
    plt.show()

def mu_law_encode(audio, quantization_channels=256):
    """Mu-law companding + quantization"""
    mu = quantization_channels - 1
    safe_audio = np.clip(audio, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(safe_audio)) / np.log1p(mu)
    signal = np.sign(safe_audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.uint8)

def mu_law_decode(encoded, quantization_channels=256):
    """Inverse mu-law decoding"""
    mu = quantization_channels - 1
    signal = 2 * (encoded.astype(np.float32) / mu) - 1
    return np.sign(signal) * (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)

def load_and_trim_audio(path, sr=16000):
    """Load and remove leading/trailing silence"""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    return trimmed

def load_nsynth_labels(json_path, audio_dir):
    """Load instrument + pitch labels for available audio files"""
    with open(json_path, "r") as f:
        metadata = json.load(f)

    file_to_label = {}
    for key, info in metadata.items():
        filename = f"{key}.wav"
        if os.path.exists(os.path.join(audio_dir, filename)):
            inst = info["instrument_family"]
            pitch = info["pitch"]
            file_to_label[filename] = (inst, pitch)

    return file_to_label

def tokenize_nsynth_with_labels(audio_dir, file_to_label, chunk_size=16000):
    """Tokenize NSynth audio into mu-law chunks with labels"""
    token_seqs = []
    labels = []

    for file, (inst, pitch) in tqdm(file_to_label.items()):
        path = os.path.join(audio_dir, file)
        audio = load_and_trim_audio(path)
        if len(audio) < chunk_size:
            continue
        for i in range(0, len(audio) - chunk_size + 1, chunk_size):
            chunk = audio[i:i + chunk_size]
            tokens = mu_law_encode(chunk)
            token_seqs.append(torch.tensor(tokens, dtype=torch.long))
            labels.append((inst, pitch))

    return token_seqs, labels

