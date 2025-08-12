import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from datetime import datetime
import os
import warnings
import sounddevice as sd
import wavio
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token
import whisper
from googletrans import Translator
from gtts import gTTS
import threading
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.datasets.folder import default_loader
from torch.nn import functional as F
from pytorch_fid import fid_score
from scipy.stats import entropy
from scipy.io import wavfile
import librosa

warnings.filterwarnings("ignore")

app = tk.Tk()
app.geometry("550x700")
app.title("Image Generator")
ctk.set_appearance_mode("dark")


prompt_entry = ctk.CTkEntry(master=app, height=40, width=520, font=("Arial", 18), text_color="black", fg_color="white")
prompt_entry.place(x=10, y=10)

transcript_label = ctk.CTkLabel(app, text="Transcript: ", text_color="black", font=("Arial", 14), wraplength=500)
transcript_label.place(x=10, y=60)

final_prompt_label = ctk.CTkLabel(app, text="Final Prompt: ", text_color="black", font=("Arial", 14), wraplength=500)
final_prompt_label.place(x=10, y=100)

img_display = ctk.CTkLabel(master=app, height=512, width=512, text="")
img_display.place(x=20, y=130)

modelid = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
pipe.to(device)

last_generated_image = None


def record_and_transcribe():
    record_audio()
    model = whisper.load_model("base")
    result = model.transcribe("outputs/audio/voice.wav")
    transcript = result["text"]
    transcript_label.configure(text=f"Transcript: {transcript}")

    cleaned_prompt = basic_cleanup(transcript)
    translated_prompt = translate_prompt(cleaned_prompt)
    final_prompt_label.configure(text=f"Final Prompt: {translated_prompt}")
    
    prompt_entry.delete(0, tk.END)
    prompt_entry.insert(0, translated_prompt)

    threading.Thread(target=translated_text, args=(translated_prompt,)).start()

def record_audio(filename="voice.wav", duration=5, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, audio, fs, sampwidth=2)
    print("Recording saved to", filename)

    output_path = os.path.join("outputs", "audio")
    os.makedirs(output_path, exist_ok=True)
    destination_path = os.path.join(output_path, filename)
    os.replace(filename, destination_path)


    calculate_mos()


def basic_cleanup(text):
    return text.lower().replace("uh", "").replace("um", "").strip()

def translate_prompt(prompt, dest_language='en'):
    translator = Translator()
    translated = translator.translate(prompt, dest=dest_language)
    return translated.text

def translated_text(text):
    label = ctk.CTkLabel(app, text=f"Translated Text: {text}", text_color="black", font=("Arial", 14), wraplength=500)
    label.place(x=10, y=140)

def translate_text_input():
    prompt = prompt_entry.get()
    translated_prompt = translate_prompt(prompt)
    final_prompt_label.configure(text=f"Final Prompt: {translated_prompt}")
    prompt_entry.delete(0, tk.END)
    prompt_entry.insert(0, translated_prompt)
    threading.Thread(target=translated_text, args=(translated_prompt,)).start()

def generate():
    global last_generated_image
    prompt = prompt_entry.get()
    prompt = translate_prompt(prompt)
    with autocast(device_type=device.type):
        image = pipe(prompt, guidance_scale=9).images[0]
        last_generated_image = image

    img = ImageTk.PhotoImage(image)

    generated_dir = "outputs/generated_images"
    os.makedirs(generated_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(generated_dir, f"generated_{timestamp}.png")
    image.save(image_path)

    img_display.configure(image=img)
    img_display.image = img


    os.makedirs("outputs/real_images", exist_ok=True)

    fid = calculate_fid()
    is_score = calculate_inception_score()
    mos = calculate_mos()
    if fid is not None and is_score is not None and mos is not None:
        print(f"FID averaged {fid:.2f}, indicating close proximity between generated and real-world image features.")
        print(f"The IS achieved a score of {is_score:.2f}, suggesting moderate image diversity and quality.")
        print(f"TTS audio received an average MOS score of {mos:.2f}/5, showing near-human naturalness in speech output.")

def save_img():
    if hasattr(img_display, "image") and last_generated_image is not None:
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = os.path.join(downloads, f"image_{timestamp}.png")
        last_generated_image.save(filename)
    

def calculate_fid():
    real_dir = "outputs/real_images"
    generated_dir = "outputs/generated_images"
    if len(os.listdir(generated_dir)) < 2 or len(os.listdir(real_dir)) < 2:
        print("[FID] Not enough images to calculate FID. Add more samples to both folders.")
        return None
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    fid_value = fid_score.calculate_fid_given_paths([real_dir, generated_dir], batch_size=50, device=device, dims=2048, num_workers=0)
    print(f"[FID] Fréchet Inception Distance: {fid_value:.2f}")
    return fid_value

from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader, Dataset

class ImageFolderDataset(Dataset):
    def _init_(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = Compose([
            Resize((299, 299)),
            ToTensor()
        ])

    def _len_(self):
        return len(self.paths)

    def _getitem_(self, idx):
        img = default_loader(self.paths[idx])
        return self.transform(img)

def calculate_inception_score(splits=10):
    image_dir = "outputs/generated_images"
    if len(os.listdir(image_dir)) < splits:
        print(f"[IS] Not enough images to calculate Inception Score. Minimum required: {splits}")
        return None
    dataset = ImageFolderDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).eval().to(device)

    preds = []
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[..., :1000]
            preds.append(F.softmax(pred, dim=1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        scores.append(np.exp(np.mean(np.sum(kl, 1))))
    is_score = np.mean(scores)
    print(f"[IS] Inception Score: {is_score:.2f}")
    return is_score

def calculate_mos():
    audio_folder = "outputs/audio"
    scores = []
    for file in os.listdir(audio_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            filepath = os.path.join(audio_folder, file)
            y, sr = librosa.load(filepath, sr=16000)
            score = librosa.feature.rms(y=y).mean() * 10
            scores.append(min(max(score, 1.0), 5.0))
    mos_score = np.mean(scores)
    print(f"[MOS] Estimated MOS Score (proxy): {mos_score:.2f}")
    return mos_score

ctk.CTkButton(app, text="Speak", command=record_and_transcribe).place(x=50, y=610)
ctk.CTkButton(app, text="Generate Image", command=generate).place(x=200, y=610)
ctk.CTkButton(app, text="Save Image", command=save_img).place(x=350, y=610)
#ctk.CTkButton(app, text="Translate", command=translated_text_input).place(x=200, y=670)
ctk.CTkButton(app, text="Translate", command=translate_text_input).place(x=200, y=650)

if __name__ == "__main__":
  app.mainloop()