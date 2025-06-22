# Refactored code that preserves original functionality, now modular and easier to maintain

import json
import os
import requests
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date

load_dotenv()

# ---------------- CONFIG ----------------

API_URL = "https://api.llama.com/v1/chat/completions"
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"
QUESTION_FILE = "questions.json"
ANSWER_FILE = "answers.json"
CHAT_LOG_FILE = "chat_log.json"

# ---------------- UTILITIES ----------------

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def normalize_type(type_str):
    return {
        "phone": "text",
        "date": "datetime",
        "string": "text"
    }.get(type_str.lower(), type_str.lower())

def load_questions(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
    
def build_system_prompt(questions):
    fields = ", ".join(questions.keys())
    return (
        f"You are a helpful assistant conducting a conversation.\n"
        f"Your goal is to ask the user about these topics: {fields}.\n"
        "Ask naturally, one at a time. When the user responds, try to determine which field it answers.\n"
        "If a field is skipped, re-ask or rephrase it later.\n"
        "At the end, summarize which questions were not answered."
    )

# ---------------- VALIDATION ----------------

def validate_answer(value, expected_type, options=None):
    expected_type = normalize_type(expected_type)
    options = options or []

    if expected_type == "text":
        return bool(value.strip())
    if expected_type == "int":
        return value.isdigit()
    if expected_type == "datetime":
        try:
            parse_date(value)
            return True
        except Exception:
            return False
    if expected_type == "boolean":
        return value.lower() in ["yes", "no"] or value.lower() in options
    if expected_type == "multichoice":
        return value.lower() in [opt.lower() for opt in options]
    return False

# ---------------- LLaMA API ----------------

def parse_response(response):
    try:
        return response["completion_message"]["content"]["text"]
    except KeyError:
        raise ValueError("Invalid response format from LLaMA API")

def answered_summary_message(answers):
    filled = {
        k: v["Value"]
        for k, v in answers.items()
        if isinstance(v, dict) and v.get("Value", "").strip()
    }
    if not filled:
        return None
    summary = "\n".join(f"{k}: {v}" for k, v in filled.items())
    return {"role": "system", "content": f"Here are the answers provided so far:\n{summary}"}

def chat_completion(conversation, answers=None):
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LLAMA_API_KEY in environment variables.")

    summary_msg = answered_summary_message(answers)
    if summary_msg:
        conversation.insert(1, summary_msg)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"model": MODEL_NAME, "messages": conversation}
    response = requests.post(API_URL, headers=headers, json=payload)

    if summary_msg:
        conversation.pop(1)

    return response.json()

# ---------------- FIELD MATCHING ----------------

def ask_llama_to_match_field(conversation, user_input, unanswered_fields, answers):
    prompt = (
        f"Based on the following unanswered fields: {', '.join(unanswered_fields)}\n"
        f"Which one does this answer refer to?\n\nUser answer: \"{user_input}\"\n"
        "Respond ONLY with the field name, exactly as written."
    )
    conversation.append({"role": "user", "content": prompt})
    match_response = chat_completion(conversation, answers)
    match_text = parse_response(match_response).strip()
    conversation.pop()

    for field in unanswered_fields:
        if match_text.lower() == field.lower():
            return field
    return None

# ---------------- ANSWER CLEANING ----------------

def extract_clean_answer(conversation, field_name, field_info, user_input, answers):
    prompt = (
        f"The user was asked about: '{field_name}'.\nExpected type: {field_info['Type']}\n"
        f"Options: {', '.join(field_info.get('Options', [])) or 'N/A'}\n"
        f"User response: \"{user_input}\"\n"
        "Respond ONLY with the cleaned value. If invalid, respond: Invalid input"
    )
    conversation.append({"role": "user", "content": prompt})
    response = chat_completion(conversation, answers)
    extracted = parse_response(response).strip()
    conversation.pop()

    if validate_answer(extracted, field_info["Type"], field_info.get("Options")):
        return extracted
    return None

# ---------------- RETRY PROMPT ----------------

def generate_retry_prompt(field_name, field_info, user_input):
    return (
        f"The user was asked to provide '{field_name}' (type: {field_info['Type']}).\n"
        f"Input: \"{user_input}\" was invalid.\n"
        "Please explain why and re-ask naturally."
    )

# ---------------- GRADIO HANDLER ----------------

def process_user_message(user_input, conversation, answers):
    exit_phrases = ["i'm done", "done answering", "no more questions", "bye", "that's all"]
    conversation.append({"role": "user", "content": user_input})

    if any(p in user_input.lower() for p in exit_phrases):
        save_json(ANSWER_FILE, answers)
        return "✅ Interview ended. Thank you for your responses!", conversation, answers

    unanswered = [k for k, v in answers.items() if v.get("Value", "").strip() == ""]
    matched_key = ask_llama_to_match_field(conversation, user_input, unanswered, answers)

    if matched_key:
        field_info = answers[matched_key]
        cleaned = extract_clean_answer(conversation, matched_key, field_info, user_input, answers)
        if cleaned:
            field_info["Value"] = cleaned
        else:
            retry_prompt = generate_retry_prompt(matched_key, field_info, user_input)
            conversation.append({"role": "user", "content": retry_prompt})
            retry_response = chat_completion(conversation, answers)
            retry_message = parse_response(retry_response).strip()
            conversation.append({"role": "assistant", "content": retry_message})
            return retry_message, conversation, answers

    response = chat_completion(conversation, answers)
    assistant_message = parse_response(response).strip()
    conversation.append({"role": "assistant", "content": assistant_message})

    # Save only when all fields (required + optional) are filled
    unanswered_all = [k for k, v in answers.items() if v.get("Value", "").strip() == ""]
    if not unanswered_all:
        save_json(ANSWER_FILE, answers)
        return "✅ All questions are answered and saved. Thank you!", conversation, answers


    return assistant_message, conversation, answers

# ---------------- MAIN ----------------

def conversation_loop():
    questions = load_json(QUESTION_FILE)
    answers = {k: v for k, v in questions.items()}
    conversation = [{"role": "system", "content": build_system_prompt(questions)}]

    print("\n🟢 Interview started. Press Enter on an empty line to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "":
            print("\n🛑 Conversation ended by user.")
            break

        msg, conversation, answers = process_user_message(user_input, conversation, answers)
        print(f"\nAssistant: {msg}")

    save_json(CHAT_LOG_FILE, conversation)
    print(f"\n💾 Chat log saved to {CHAT_LOG_FILE}")

if __name__ == "__main__":
    conversation_loop()
