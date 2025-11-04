import gradio as gr
import assemblyai as aai
from translate import Translator
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play 
import uuid
from pathlib import Path 
load_dotenv()

def voice_to_voice(audio_file):
    transcription_response=audio_transcription(audio_file)
    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    ar_translation,zh_translation,ur_translation=text_translation(text)
    ar_audio_path=text_to_speech(ar_translation)
    ar_path=Path(ar_audio_path)
    zh_audio_path=text_to_speech(zh_translation)
    zh_path=Path(zh_audio_path)
    ur_audio_path=text_to_speech(ur_translation)
    ur_path=Path(ur_audio_path)
    
    return ar_path,zh_path,ur_path

def audio_transcription(audio_file):
    aai.settings.api_key = "" # replace with a valid API key
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    return transcription


def text_translation(text):
    translator_ar=Translator(from_lang='en',to_lang="ar")
    ar_text=translator_ar.translate(text)

    translator_zh=Translator(from_lang='en',to_lang="zh")
    zh_text=translator_zh.translate(text)
    translator_ur=Translator(from_lang='en',to_lang="ur")
    ur_text=translator_ur.translate(text)
    return ar_text,zh_text,ur_text
 
def text_to_speech(text):
   
    client = ElevenLabs(api_key="") # replace with a valid API key

    audio = client.text_to_speech.convert(
    text = text,
    voice_id = "",      # replace with a voice ID available for you
    model_id = "eleven_multilingual_v2",   # or another supported model
    output_format = "mp3_44100_128"
)

    save_file_path = f"{uuid.uuid4()}.mp3"

    # Write each chunk to the file
    with open(save_file_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")
    return save_file_path

audio_input = gr.Audio(sources=["microphone"], type="filepath")

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[gr.Audio(label="Arabic"), gr.Audio(label="Chinese"), gr.Audio(label="Urdu")])
if __name__ == "__main__":

    demo.launch()