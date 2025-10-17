# ================================================================
# 🎧 EchoVerse – IBM Granite version
# ================================================================

# 1️⃣ Install dependencies
!pip install torch torchvision torchaudio -q
!pip install accelerate transformers gTTS ipywidgets -q

# 2️⃣ Imports
from transformers import pipeline
from gtts import gTTS
from IPython.display import Audio, display, HTML
import ipywidgets as widgets
import tempfile, os

# 3️⃣ Load IBM Granite model
# WARNING: This is a 2B parameter model, make sure your Colab runtime is GPU
model_name = "ibm-granite/granite-3.3-2b-instruct"
print("Loading IBM Granite model (this may take a few minutes)...")
generator = pipeline("text-generation", model=model_name, device_map="auto")  # GPU recommended

# 4️⃣ Tone templates
TONE_PROMPTS = {
    "Neutral": "Rewrite the following text in a calm, neutral, clear tone without losing its meaning.",
    "Suspenseful": "Rewrite the following text to sound suspenseful, thrilling, and tense while keeping the facts the same.",
    "Inspiring": "Rewrite the following text in an inspiring, motivational tone that uplifts the reader while preserving the meaning."
}

def build_prompt(original_text, tone):
    instr = TONE_PROMPTS.get(tone, TONE_PROMPTS["Neutral"])
    return f"{instr}\n\nOriginal:\n{original_text}\n\nRewritten:"

# 5️⃣ Generate text using IBM Granite
def granite_generate(prompt, max_length=300):
    # The IBM Granite model expects a "messages" style input like ChatGPT
    messages = [{"role": "user", "content": prompt}]
    output = generator(messages, max_new_tokens=max_length)
    # Extract generated text
    # Ensure the output is a string before returning
    text = ""
    if output and isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
        text = output[0]['generated_text']
    else:
        text = str(output) # Fallback to string representation if format is unexpected

    if "Rewritten:" in text:
        text = text.split("Rewritten:")[-1].strip()
    return text


# 6️⃣ Text-to-speech
def text_to_speech(text, filename="echoverse_output.mp3"):
    tts = gTTS(text)
    path = os.path.join(tempfile.gettempdir(), filename)
    tts.save(path)
    return path

# 7️⃣ UI widgets
upload = widgets.FileUpload(accept='.txt', multiple=False)
text_area = widgets.Textarea(
    placeholder="Paste text here or upload a .txt file",
    layout=widgets.Layout(width='100%', height='220px')
)
tone_selector = widgets.Dropdown(
    options=["Neutral", "Suspenseful", "Inspiring"],
    value="Neutral",
    description="Tone:"
)
status = widgets.Output()
generate_btn = widgets.Button(description="🎧 Generate Audiobook", button_style='success')
download_out = widgets.Output()

# 8️⃣ Handle file upload
def on_file_upload(change):
    if upload.value:
        for fn, fileinfo in upload.value.items():
            try:
                text_area.value = fileinfo['content'].decode('utf-8')
            except Exception:
                text_area.value = fileinfo['content'].decode('latin-1')
            break
upload.observe(on_file_upload, names='value')

# 9️⃣ Generate button callback
def on_generate_clicked(b):
    download_out.clear_output()
    status.clear_output()
    with status:
        print("Processing... this may take a few minutes ⏳")
    text = text_area.value.strip()
    if not text:
        with status:
            print("⚠️ Please paste text or upload a file first.")
        return
    tone = tone_selector.value
    prompt = build_prompt(text, tone)

    # 1. Rewrite
    with status:
        print(f"Generating {tone.lower()} version using IBM Granite...")
    rewritten_output = granite_generate(prompt)

    # 2. Convert to speech
    with status:
        print("Converting to speech...")
    # Explicitly ensure rewritten_output is a string before passing to text_to_speech
    if not isinstance(rewritten_output, str):
        rewritten_text = str(rewritten_output)
    else:
        rewritten_text = rewritten_output

    mp3_path = text_to_speech(rewritten_text)

    # 3. Show results
    with status:
        print("✅ Done! Listen below or download the MP3 file.")
        display(Audio(mp3_path, autoplay=False))
    with download_out:
        display(HTML(f'<a href="/files{mp3_path}" download>⬇️ Download MP3</a>'))

generate_btn.on_click(on_generate_clicked)

# 10️⃣ Display UI
display(widgets.VBox([
    widgets.HTML("<h3>🎧 EchoVerse — IBM Granite Audiobook Creator</h3>"),
    widgets.HTML("<p>Upload or paste text, choose tone, then click 'Generate Audiobook'</p>"),
    upload,
    tone_selector,
    text_area,
    generate_btn,
    status,
    download_out
]))