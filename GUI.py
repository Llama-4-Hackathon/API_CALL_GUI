import gradio as gr
import json
import os
from API_Call1 import process_user_message, load_questions, build_system_prompt

# Paths
ANSWER_FILE = "answers.json"
CHAT_LOG_FILE = "chat_log.json"

# Load initial state
questions = load_questions("questions.json")
initial_answers = {k: v for k, v in questions.items()}
initial_conversation = [{"role": "system", "content": build_system_prompt(questions)}]

# Chat logic
def chat_interface(user_input, history, conversation, answers):
    reply, conversation, answers = process_user_message(user_input, conversation, answers)
    history = history or []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return "", history, conversation, answers

# Read saved answers
def load_answers():
    if os.path.exists(ANSWER_FILE):
        with open(ANSWER_FILE, "r") as f:
            return json.dumps(json.load(f), indent=2)
    return "No saved answers yet."

# Read chat log
def load_chat_history():
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r") as f:
            return json.dumps(json.load(f), indent=2)
    return "No chat history found."

# Reset
def reset_conversation():
    new_questions = load_questions("questions.json")
    new_answers = {k: v for k, v in new_questions.items()}
    new_conversation = [{"role": "system", "content": build_system_prompt(new_questions)}]
    return "", [], new_conversation, new_answers

# GUI Layout
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Form-Filling Assistant Chatbot")

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(label="Conversation", type="messages")
            state_conv = gr.State(initial_conversation)
            state_ans = gr.State(initial_answers)

            with gr.Row():
                msg = gr.Textbox(placeholder="Type your message here...", scale=4, show_label=False)
                send = gr.Button("Send", scale=1)

            send.click(
                fn=chat_interface,
                inputs=[msg, chatbot, state_conv, state_ans],
                outputs=[msg, chatbot, state_conv, state_ans]
            )
            msg.submit(
                fn=chat_interface,
                inputs=[msg, chatbot, state_conv, state_ans],
                outputs=[msg, chatbot, state_conv, state_ans]
            )

            with gr.Row():
                reset = gr.Button("🔄 Reset Form")
                reset.click(
                    fn=reset_conversation,
                    outputs=[msg, chatbot, state_conv, state_ans]
                )

        with gr.Tab("📄 Saved Answers"):
            gr.Markdown("### ✅ Current Form Answers")
            answer_display = gr.Code(label="answers.json", language="json")
            load_btn = gr.Button("🔄 Refresh Answers")
            load_btn.click(fn=load_answers, outputs=answer_display)

            gr.File(value=ANSWER_FILE, label="📥 Download answers.json", interactive=True)

        with gr.Tab("🕓 Chat Log"):
            gr.Markdown("### 📝 Previous Conversation Log")
            history_display = gr.Code(label="chat_log.json", language="json")
            history_btn = gr.Button("🔄 Load Chat History")
            history_btn.click(fn=load_chat_history, outputs=history_display)

demo.launch()
