import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading
import torch
import torchaudio
import os
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from speechbrain.pretrained import EncoderClassifier

model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

speaker_db = {}
DB_FILE = "speakers_db.pt"

def get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    emb = model.encode_batch(signal)
    emb = torch.nn.functional.normalize(emb, dim=1)  
    return emb.mean(dim=1)

def record_audio(filename, duration=5, fs=16000):
    messagebox.showinfo("Recording", f"🎤 Speak now ({duration} sec)...")
    num_samples = int(duration * fs)
    recording = sd.rec(num_samples, samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    write(filename, fs, (recording * 32767).astype(np.int16))

def register_speaker():
    name = simpledialog.askstring("Register Speaker", "Enter speaker name:")
    if not name:
        return
    threading.Thread(target=register_from_mic, args=(name,)).start()

def register_from_mic(name):
    embeddings = []
    for i in range(3):
        filepath = f"{name}_{i}.wav"
        record_audio(filepath, duration=5)
        emb = get_embedding(filepath)
        embeddings.append(emb)

    final_emb = torch.mean(torch.stack(embeddings), dim=0)

    if name in speaker_db:
        speaker_db[name] = (speaker_db[name] + final_emb) / 2
    else:
        speaker_db[name] = final_emb

    update_speaker_list()
    save_db()
    messagebox.showinfo("Success", f"✅ Speaker '{name}' registered with 3 samples!")

def recognize_speaker():
    if not speaker_db:
        messagebox.showwarning("Warning", "⚠️ No registered speakers yet!")
        return
    threading.Thread(target=recognize_from_mic).start()

def recognize_from_mic():
    filepath = "test.wav"
    record_audio(filepath, duration=5)
    test_emb = get_embedding(filepath)

    similarities = {}
    for name, emb in speaker_db.items():
        sim = F.cosine_similarity(test_emb, emb).item()
        similarities[name] = sim

    best_match = max(similarities, key=similarities.get)
    confidence = similarities[best_match]

    if confidence > 0.4:
        messagebox.showinfo("Recognition Result", f"🎤 Speaker: {best_match}\nConfidence: {confidence:.2f}")
    else:
        messagebox.showinfo("Recognition Result", f"⚠️ Unknown Speaker\n(Max confidence: {confidence:.2f})")

def delete_speaker():
    selection = speakers_listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "⚠️ Please select a speaker to delete.")
        return

    name = speakers_listbox.get(selection[0])
    confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{name}'?")
    if confirm:
        del speaker_db[name]
        update_speaker_list()
        save_db()
        messagebox.showinfo("Deleted", f"🗑️ Speaker '{name}' removed successfully.")

def save_db():
    torch.save(speaker_db, DB_FILE)

def load_db():
    global speaker_db
    if os.path.exists(DB_FILE):
        speaker_db = torch.load(DB_FILE)
        update_speaker_list()

def update_speaker_list():
    speakers_listbox.delete(0, tk.END)
    for name in speaker_db.keys():
        speakers_listbox.insert(tk.END, name)

root = tk.Tk()
root.title("🎙️ Realtime Speaker Recognition")
root.geometry("500x450")

title_label = ttk.Label(root, text="Realtime Speaker Recognition", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)

register_btn = ttk.Button(btn_frame, text="➕ Register Speaker", command=register_speaker)
register_btn.grid(row=0, column=0, padx=10)

recognize_btn = ttk.Button(btn_frame, text="🔍 Recognize Speaker", command=recognize_speaker)
recognize_btn.grid(row=0, column=1, padx=10)

delete_btn = ttk.Button(btn_frame, text="🗑️ Delete Speaker", command=delete_speaker)
delete_btn.grid(row=0, column=2, padx=10)

quit_btn = ttk.Button(btn_frame, text="❌ Quit", command=root.quit)
quit_btn.grid(row=0, column=3, padx=10)

list_label = ttk.Label(root, text="Registered Speakers:")
list_label.pack(pady=5)

speakers_listbox = tk.Listbox(root, height=10, font=("Arial", 12))
speakers_listbox.pack(fill="both", expand=True, padx=20, pady=5)

load_db()

root.mainloop()
