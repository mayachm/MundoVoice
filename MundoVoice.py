import os
import gradio as gr
import assemblyai as aai
from deep_translator import GoogleTranslator
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from tempfile import NamedTemporaryFile

# ✅ set your keys
os.environ["ASSEMBLYAI_API_KEY"] = "" 
os.environ["ELEVENLABS_API_KEY"] = ""

# ✅ init clients
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# ✅ language to voice id mapping
VOICE_IDS = {
    "en": "kPzsL2i3teMYv0FxEYQ6", # get your voice ids from eleven labs
    "ar": "B5xxC4eQoOFJnY4R5XkI",
    "zh-CN":"kPzsL2i3teMYv0FxEYQ6",  
    "ur": "XzM6ifF57Bo5L0jHJHrz",
}

# ✅ Correct translator language codes
LANG_CODES = {
    "en": "en",
    "ar": "ar",
    "zh": "zh-CN",
    "zh-CN": "zh-CN",
    "ur": "ur",
}

# 🔊 Transcribe audio
def audio_transcription(audio_file, source_lang):
    lang = LANG_CODES.get(source_lang, "en")

    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(
        audio_file,
        config=aai.TranscriptionConfig(language_code=lang)
    )
    return transcription.text

# 🌍 Translate text
def translate_text(text, source_lang, target_lang):
    src = LANG_CODES.get(source_lang, "en")
    tgt = LANG_CODES.get(target_lang, "en")

    if src == tgt:
        return text

    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except Exception as e:
        return f"Translation Error: {e}"

# 🔈 TTS
def text_to_speech(text, lang):
    vid = VOICE_IDS.get(lang, VOICE_IDS["en"])

    audio_stream = client.text_to_speech.convert(
        voice_id=vid,
        model_id="eleven_multilingual_v2",
        text=text
    )

    with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        for chunk in audio_stream:
            if chunk:
                temp_audio.write(chunk)
        return temp_audio.name

# 🎧 Full pipeline
def process_audio(audio_file, source_lang, target_lang):
    if not audio_file:
        return "No audio provided.", None

    try:
        text = audio_transcription(audio_file, source_lang)
        translated = translate_text(text, source_lang, target_lang)
        output_audio = text_to_speech(translated, target_lang)
        return translated, output_audio
    except Exception as e:
        return f"Error: {e}", None

# 🎨 UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌐 Meet MundoVoice")

    with gr.Row():
        source_lang = gr.Dropdown(["en", "ar", "zh-CN", "ur"], label="From", value="en")
        target_lang = gr.Dropdown(["en", "ar", "zh-CN", "ur"], label="To", value="ar")

    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Record voice")
    translate_button = gr.Button("Translate 🎧")
    output_text = gr.Textbox(label="📝 Translated Text")
    output_audio = gr.Audio(label="🔊 Translated Voice")

    translate_button.click(
        fn=process_audio,
        inputs=[audio_input, source_lang, target_lang],
        outputs=[output_text, output_audio]
    )

demo.launch()
