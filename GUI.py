import os
import gradio as gr
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from API_Call1 import (
    list_forms_in_folder,
    load_json,
    save_json,
    build_system_prompt,
    process_user_message
)
# Voice imports
from speechtotext import load_model, transcribe_audio

# --- Paths ----
FORM_FOLDER = "API_CALL_GUI/Forms"
OUTPUT_FOLDER = "output"
CHAT_LOG_FILE = "chat_log.json"

# --- Global Session State ---
session = {
    "conversation": [],
    "answers": {},
    "form_name": None,
    "answer_file_path": None
}

# Load Whisper model once for voice input
audio_model = load_model()

# --- Utility Functions ---

def generate_pdf_from_answers(json_data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    text = c.beginText(40, height - 50)
    text.setFont("Helvetica", 12)
    text.textLine("📝 Form Responses")
    text.textLine("-----------------------------")

    for key, value in json_data.items():
        val = value.get("Value", "N/A")
        text.textLine(f"{key}: {val}")

    c.drawText(text)
    c.save()

# --- Form Initialization ---

def initialize_session():
    form_files = list_forms_in_folder(FORM_FOLDER)
    form_names = [os.path.splitext(f)[0].replace("_", " ") for f in form_files]
    return form_names


def start_form(selected_form):
    form_file = f"{selected_form.replace(' ', '_')}.json"
    form_path = os.path.join(FORM_FOLDER, form_file)

    if not os.path.exists(form_path):
        return [{"role": "assistant", "content": f"❌ Could not find the form: {selected_form}"}], "", None, None

    session["form_name"] = selected_form
    session["answers"] = load_json(form_path)
    session["conversation"] = [{
        "role": "system",
        "content": build_system_prompt(session["answers"])
    }]
    session["answer_file_path"] = os.path.join(OUTPUT_FOLDER, f"{selected_form.replace(' ', '_')}_answers.json")

    first_field = list(session["answers"].keys())[0]
    session["conversation"].append({
        "role": "assistant",
        "content": f"Let's begin. Could you tell me your {first_field.lower()}?"
    })

    return session["conversation"], f"📝 Started form: {selected_form}", None, None

# --- Chat & Voice Interaction ---

def chat_interface(text_input, chat_history):
    if not text_input.strip():
        return text_input, chat_history

    msg, updated_conv, updated_answers, done = process_user_message(
        text_input,
        session["conversation"],
        session["answers"],
        session["answer_file_path"]
    )

    session["conversation"] = updated_conv
    session["answers"] = updated_answers

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": text_input})
    chat_history.append({"role": "assistant", "content": msg})

    if done:
        save_json(session["answer_file_path"], session["answers"])
        save_json(CHAT_LOG_FILE, session["conversation"])
        pdf_path = session["answer_file_path"].replace(".json", ".pdf")
        generate_pdf_from_answers(session["answers"], pdf_path)
        # expose downloads via gr.File
        return "", chat_history, gr.update(value=session["answer_file_path"], visible=True), gr.update(value=pdf_path, visible=True)

    return "", chat_history, gr.update(visible=False), gr.update(visible=False)


def voice_interface(audio_file, chat_history):
    if not audio_file:
        return "", chat_history or []
    text = transcribe_audio(audio_model, audio_file)
    if not text:
        return "", chat_history
    return chat_interface(text, chat_history)

# --- UI Setup ---

with gr.Blocks() as demo:
    gr.Markdown("## 📝 AI-Powered Form Filler with Voice Input")

    with gr.Row():
        form_selector = gr.Dropdown(label="Select a form", choices=initialize_session(), interactive=True)
        start_btn = gr.Button("Start Conversation")

    chatbox = gr.Chatbot(label="Conversation", type="messages")
    text_input = gr.Textbox(label="Your message", placeholder="Type your response and press Enter")
    send_btn = gr.Button("Send")

    # Voice recorder (default dual-mode; record and upload)
    mic = gr.Audio(
        type="filepath",
        label="🎤 Record your response"
    )
    voice_btn = gr.Button("Record & Send")

    with gr.Row():
        download_json = gr.File(label="⬇️ Download JSON", visible=False)
        download_pdf = gr.File(label="⬇️ Download PDF", visible=False)

    with gr.Row():
        clear_btn = gr.Button("Clear Chat")
        status = gr.Markdown()

    # Bindings
    start_btn.click(start_form, inputs=form_selector,
                    outputs=[chatbox, status, download_json, download_pdf])

    send_btn.click(chat_interface,
                   inputs=[text_input, chatbox],
                   outputs=[text_input, chatbox, download_json, download_pdf])
    text_input.submit(chat_interface,
                      inputs=[text_input, chatbox],
                      outputs=[text_input, chatbox, download_json, download_pdf])

    voice_btn.click(voice_interface,
                    inputs=[mic, chatbox],
                    outputs=[text_input, chatbox, download_json, download_pdf])

    clear_btn.click(lambda: ("", [], gr.update(visible=False), gr.update(visible=False)),
                    outputs=[text_input, chatbox, download_json, download_pdf])

# --- Launch ---
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    demo.launch()

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    demo.launch()
