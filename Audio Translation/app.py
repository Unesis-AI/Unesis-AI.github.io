import os
import numpy as np
import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path


def voice_to_voice(audio_file):

    # Transcript speech
    transcript = transcribe_audio(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        raise gr.Error(transcript.error)
    
    transcript_text = transcript.text

    # Translate text
    list_translations = translate_text(transcript_text)
    generated_audio_paths = []

    # Generate speech from text
    for translation in list_translations:
        translated_audio_file_name = text_to_speech(translation)
        path = Path(translated_audio_file_name)
        generated_audio_paths.append(path)

    return (*generated_audio_paths, *list_translations)


# Function to transcribe audio using AssemblyAI
def transcribe_audio(audio_file):
    aai.settings.api_key = "d6006b974a26450e92989864799686c4"

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    return transcript

    
# Function to translate text
def translate_text(text: str) -> list:
    languages = ["es", "rw", "sw"]
    list_translations = []

    for lan in languages:
        translator = Translator(from_lang="en", to_lang=lan)
        translation = translator.translate(text)
        list_translations.append(translation)

    return list_translations

# Function to generate speech
def text_to_speech(text: str) -> str:
    client = ElevenLabs(
        api_key="sk_ffeeeea80a6979eff91ae526330735c3363e98b70153db79",
    )

    response = client.text_to_speech.convert(
        voice_id="ZF6FPAbjXT4488VcRRnw",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    save_file_path = f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    return save_file_path


input_audio = gr.Audio(
    sources=["microphone"],
    type="filepath",
    show_download_button=True,
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)

with gr.Blocks() as demo:
    gr.Markdown("## Welcome to Unesis Voice Translator")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"],
                                type="filepath",
                                show_download_button=True,
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#01C6FF",
                                    waveform_progress_color="#0066B4",
                                    skip_length=2,
                                    show_controls=False,
                                ),)
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                btn = gr.ClearButton(audio_input, "Clear")

    with gr.Row():
        with gr.Group() as spanish:
            es_output = gr.Audio(label="Spanish", interactive=False)
            es_text = gr.Markdown()

        with gr.Group() as kinyarwanda:
            rw_output = gr.Audio(label="Kinyarwanda", interactive=False)
            rw_text = gr.Markdown()

        with gr.Group() as swahili:
            sw_output = gr.Audio(label="Swahili", interactive=False)
            sw_text = gr.Markdown()

    output_components = [es_output, rw_output, sw_output, es_text, rw_text, sw_text]
    submit.click(fn=voice_to_voice, inputs=audio_input, outputs=output_components, show_progress=True)

if __name__ == "__main__":
    demo.launch()