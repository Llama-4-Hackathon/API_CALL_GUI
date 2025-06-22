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

# ---------------- UTILS ----------------

def load_questions(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def build_system_prompt(questions_dict):
    fields = ", ".join(questions_dict.keys())
    return (
        "You are a helpful assistant conducting a conversation.\n"
        f"Your goal is to ask the user about these topics: {fields}.\n"
        "Ask naturally, one at a time. When the user responds, try to determine which field it answers.\n"
        "If a field is skipped, re-ask or rephrase it later.\n"
        "At the end, summarize which questions were not answered."
    )

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
    summary = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in filled.items())
    return {"role": "system", "content": f"Here are the answers the user has provided so far:\n{summary}"}

def chat_completion(conversation, answers=None):
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LLAMA_API_KEY in environment variables.")

    summary_msg = answered_summary_message(answers) if answers else None
    if summary_msg:
        conversation.insert(1, summary_msg)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": conversation
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if summary_msg:
        conversation.pop(1)

    return response.json()

# ---------------- MATCHING ----------------

def ask_llama_to_match_field(conversation, user_input, unanswered_fields, answers):
    prompt = (
        "Based on the following unanswered fields: "
        f"{', '.join(unanswered_fields)}\n"
        f"Which one does this answer refer to?\n\nUser answer: \"{user_input}\"\n"
        "Respond ONLY with the field name, exactly as written."
    )
    conversation.append({"role": "user", "content": prompt})
    match_response = chat_completion(conversation, answers)
    match_text = parse_response(match_response).strip()
    conversation.pop()

    # Match field ignoring case
    for field in unanswered_fields:
        if match_text.lower() == field.lower():
            return field
    return None

# ---------------- VALIDATION ----------------

def validate_answer(value, expected_type, options=None):
    options = options or []
    if expected_type == "text":
        return bool(value.strip())
    elif expected_type == "int":
        return value.isdigit()
    elif expected_type == "datetime":
        try:
            parse_date(value)
            return True
        except Exception:
            return False
    elif expected_type == "boolean":
        return value.lower() in ["yes", "no"] or value.lower() in options
    elif expected_type == "multichoice":
        return value.lower() in [opt.lower() for opt in options]
    return False

# ---------------- CLEANING ----------------

def extract_clean_answer(conversation, field_name, field_info, user_input, answers):
    prompt = (
        f"The user was asked the question related to the field: '{field_name}'.\n"
        f"Expected type: {field_info['Type']}.\n"
        f"Available options: {', '.join(field_info.get('Options', [])) or 'N/A'}.\n"
        f"User response: \"{user_input}\"\n"
        "Your task is to extract only the valid and concise value for this field.\n"
        "⚠️ Respond with ONLY the extracted value. Do not include any explanations, greetings, or follow-up questions.\n"
        "If the input is unclear or invalid, respond with exactly: Invalid input"
    )
    conversation.append({"role": "user", "content": prompt})
    response = chat_completion(conversation, answers)
    extracted = parse_response(response).strip()
    conversation.pop()

    is_valid = validate_answer(extracted, field_info["Type"], field_info.get("Options"))
    return extracted if is_valid else None

# ---------------- RETRY ----------------

def generate_retry_prompt(field_name, field_info, user_input):
    return (
        f"The user was asked to provide their '{field_name}'.\n"
        f"Expected type: {field_info['Type']}.\n"
        f"Options (if applicable): {', '.join(field_info.get('Options', [])) or 'N/A'}.\n"
        f"Their input was: \"{user_input}\"\n"
        "This input is invalid. Please politely explain why and ask the question again naturally."
    )

# ---------------- CLI LOOP ----------------

def conversation_loop():
    questions = load_questions(QUESTION_FILE)
    answers = {k: v for k, v in questions.items()}
    save_json(ANSWER_FILE, answers)

    system_prompt = build_system_prompt(questions)
    conversation = [{"role": "system", "content": system_prompt}]

    print("\n🟢 Interview started. Press Enter on an empty line to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "":
            print("\n🛑 Conversation ended by user.")
            break

        conversation.append({"role": "user", "content": user_input})

        unanswered = [k for k, v in answers.items() if v.get("Value", "").strip() == ""]
        exit_phrases = ["i'm done", "done answering", "no more questions", "bye", "that's all"]
        if any(p in user_input.lower() for p in exit_phrases):
            print("\n🛑 You indicated you're finished. Wrapping up the interview.")
            break

        matched_key = ask_llama_to_match_field(conversation, user_input, unanswered, answers)

        if matched_key:
            field_info = answers[matched_key]
            cleaned = extract_clean_answer(conversation, matched_key, field_info, user_input, answers)

            if cleaned is None:
                retry_prompt = generate_retry_prompt(matched_key, field_info, user_input)
                conversation.append({"role": "user", "content": retry_prompt})
                retry_response = chat_completion(conversation, answers)
                retry_message = parse_response(retry_response).strip()
                conversation.append({"role": "assistant", "content": retry_message})
                print(f"\nAssistant: {retry_message}")
                continue

            field_info["Value"] = cleaned
            save_json(ANSWER_FILE, answers)
            print(f"[✔] Saved: {matched_key} = {cleaned}")

        response = chat_completion(conversation, answers)
        assistant_message = parse_response(response).strip()
        conversation.append({"role": "assistant", "content": assistant_message})
        print(f"\nAssistant: {assistant_message}")

        required_fields = [k for k, v in answers.items() if v.get("Required") and v.get("Value", "").strip() == ""]
        if not required_fields:
            print("\n✅ All required fields answered!")
            break

    save_json(ANSWER_FILE, answers)
    save_json(CHAT_LOG_FILE, conversation)
    print(f"\n💾 Chat log saved to {CHAT_LOG_FILE}")
    print(f"💾 Answers saved to {ANSWER_FILE}")

# ---------------- GRADIO HOOK ----------------

questions = load_questions(QUESTION_FILE)
answers = {k: v for k, v in questions.items()}
conversation = [{"role": "system", "content": build_system_prompt(questions)}]

def process_user_message(user_input):
    exit_phrases = ["i'm done", "done answering", "no more questions", "bye", "that's all"]
    if any(p in user_input.lower() for p in exit_phrases):
        return "✅ Interview ended. Thank you for your responses!"

    conversation.append({"role": "user", "content": user_input})
    unanswered = [k for k, v in answers.items() if v.get("Value", "").strip() == ""]

    matched_key = ask_llama_to_match_field(conversation, user_input, unanswered, answers)

    if matched_key:
        field_info = answers[matched_key]
        cleaned = extract_clean_answer(conversation, matched_key, field_info, user_input, answers)

        if cleaned is None:
            retry_prompt = generate_retry_prompt(matched_key, field_info, user_input)
            conversation.append({"role": "user", "content": retry_prompt})
            retry_response = chat_completion(conversation, answers)
            retry_message = parse_response(retry_response).strip()
            conversation.append({"role": "assistant", "content": retry_message})
            return retry_message

        field_info["Value"] = cleaned
        save_json(ANSWER_FILE, answers)

    response = chat_completion(conversation, answers)
    assistant_message = parse_response(response).strip()
    conversation.append({"role": "assistant", "content": assistant_message})

    return assistant_message

# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    conversation_loop()
