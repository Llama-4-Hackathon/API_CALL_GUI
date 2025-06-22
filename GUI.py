import gradio as gr
from API_Call import (
    list_forms_in_folder,
    load_json,
    save_json,
    build_system_prompt,
    process_user_message
)
import os

# Paths
FORM_FOLDER = "forms"
OUTPUT_FOLDER = "output"
CHAT_LOG_FILE = "chat_log.json"

# Global session state
session = {
    "conversation": [],
    "answers": {},
    "form_name": None,
    "answer_file_path": None
}


def initialize_session():
    # Load available forms
    form_files = list_forms_in_folder(FORM_FOLDER)
    form_names = [os.path.splitext(f)[0].replace("_", " ") for f in form_files]
    return form_names


def start_form(selected_form):
    form_file = f"{selected_form.replace(' ', '_')}.json"
    form_path = os.path.join(FORM_FOLDER, form_file)
    if not os.path.exists(form_path):
        return [{"role": "assistant", "content": f"❌ Could not find the form: {selected_form}"}], ""

    session["form_name"] = selected_form
    session["answers"] = load_json(form_path)
    session["conversation"] = [{
        "role": "system",
        "content": build_system_prompt(session["answers"])
    }]
    session["answer_file_path"] = os.path.join(OUTPUT_FOLDER, f"{selected_form.replace(' ', '_')}_answers.json")

    # Start the conversation with first question
    first_field = list(session["answers"].keys())[0]
    session["conversation"].append({
        "role": "assistant",
        "content": f"Let's begin. Could you tell me your {first_field.lower()}?"
    })

    return session["conversation"], f"📝 Started form: {selected_form}"


def chat_interface(user_input, chat_history):
    if not user_input.strip():
        return chat_history, ""

    msg, updated_conv, updated_answers, done = process_user_message(
        user_input,
        session["conversation"],
        session["answers"],
        session["answer_file_path"]
    )

    session["conversation"] = updated_conv
    session["answers"] = updated_answers

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": msg})

    if done:
        save_json(session["answer_file_path"], session["answers"])
        save_json(CHAT_LOG_FILE, session["conversation"])

    return chat_history, ""


# UI setup
with gr.Blocks() as demo:
    gr.Markdown("## 📝 AI-Powered Form Filler")

    with gr.Row():
        form_selector = gr.Dropdown(label="Select a form", choices=initialize_session(), interactive=True)
        start_btn = gr.Button("Start Conversation")

    chatbox = gr.Chatbot(label="Conversation", type="messages")
    user_input = gr.Textbox(label="Your message")
    clear_btn = gr.Button("Clear Chat")

    status = gr.Markdown()

    # Events
    start_btn.click(fn=start_form, inputs=form_selector, outputs=[chatbox, status])
    user_input.submit(fn=chat_interface, inputs=[user_input, chatbox], outputs=[chatbox, user_input])
    clear_btn.click(lambda: ([], ""), outputs=[chatbox, user_input])

# Launch
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    demo.launch()
