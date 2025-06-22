import os
import json
import requests
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date
import re

load_dotenv()

# ---------------- CONFIG ----------------

API_URL = "https://api.llama.com/v1/chat/completions"
MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"
FORM_FOLDER = "./Forms"
OUTPUT_FOLDER = "output"
CHAT_LOG_FILE = "chat_log.json"

# ---------------- UTILITIES ----------------

def clean_value(raw_value, expected_type):
    if raw_value is None:
        return None

    val = raw_value.strip().lower()

    if expected_type == "boolean":
        if any(p in val for p in ["yes", "yeah", "yep", "true", "of course"]):
            return "yes"
        if any(p in val for p in ["no", "nope", "nah", "false"]):
            return "no"

    if expected_type == "int":
        try:
            return str(int(float(val.replace(",", ""))))
        except ValueError:
            return None

    if expected_type == "float":
        try:
            val = val.replace(",", "").replace("$", "")
            return str(float(val))
        except ValueError:
            return None

    return raw_value.strip()


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


def list_forms_in_folder(folder_path="./Forms"):
    return sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".json") and not f.startswith(".")
    ])


# ---------------- VALIDATION ----------------

def validate_answer(value, expected_type, options=None):
    expected_type = normalize_type(expected_type)
    cleaned = clean_value(value, expected_type)

    if cleaned is None:
        return False

    if expected_type == "text":
        return bool(cleaned)

    if expected_type == "int":
        return cleaned.isdigit()

    if expected_type == "float":
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    if expected_type == "datetime":
        try:
            parse_date(cleaned)
            return True
        except Exception:
            return False

    if expected_type == "boolean":
        return cleaned in ["yes", "no"]

    if expected_type == "multichoice":
        return cleaned.lower() in [opt.lower() for opt in options or []]

    return False

# ---------------- LLaMA API ----------------

def parse_response(response):
    try:
        return response["completion_message"]["content"]["text"]
    except KeyError:
        raise ValueError("Invalid response format from LLaMA API")

def answered_summary_message(answers):
    if not answers:
        return None
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
    # Skip matching if input is too short or non-informative
    if len(user_input.strip()) < 3 or user_input.lower() in ["hi", "hello", "let’s start", "start"]:
        return None

    prompt = (
        f"Based on the following unanswered fields: {', '.join(unanswered_fields)}\n"
        f"Which one does this answer refer to?\n\nUser answer: \"{user_input}\"\n"
        "Respond ONLY with the field name, exactly as written. If unsure, say: None"
    )

    conversation.append({"role": "user", "content": prompt})
    match_response = chat_completion(conversation, answers)
    match_text = parse_response(match_response).strip()
    conversation.pop()

    # Explicitly check for None or invalid
    for field in unanswered_fields:
        if match_text.lower() == field.lower():
            return field
    return None

# ---------------- ANSWER CLEANING ----------------

def extract_clean_answer(conversation, field_name, field_info, user_input, answers):
    expected_type = field_info["Type"]

    # Try local cleaning first (avoid LLM when we can)
    cleaned = clean_value(user_input, expected_type)
    if validate_answer(cleaned, expected_type, field_info.get("Options")):
        return cleaned

    # Otherwise ask LLM to clean it strictly
    prompt = (
        f"Extract ONLY the value for this form field.\n"
        f"Field: {field_name}\n"
        f"Expected Type: {expected_type}\n"
        f"Options: {', '.join(field_info.get('Options', [])) or 'N/A'}\n"
        f"User Input: \"{user_input}\"\n"
        f"Rules:\n"
        f"- Do NOT include any explanation, emoji, or commentary.\n"
        f"- Do NOT add currency symbols, commas, or extra words.\n"
        f"- Just return the raw number or answer.\n"
        f"- If the input is clearly invalid, return exactly: Invalid input"
    )

    conversation.append({"role": "user", "content": prompt})
    response = chat_completion(conversation, answers)
    extracted = parse_response(response).strip()
    conversation.pop()

    cleaned = clean_value(extracted, expected_type)
    if validate_answer(cleaned, expected_type, field_info.get("Options")):
        return cleaned

    return None


# ---------------- RETRY PROMPT ----------------

def generate_retry_prompt(field_name, field_info, user_input):
    return (
        f"The user was asked to provide '{field_name}' (type: {field_info['Type']}).\n"
        f"Input: \"{user_input}\" was invalid.\n"
        "Please explain why and re-ask naturally."
    )

# ---------------- FORM SELECTION VIA LLM ----------------

def select_form_via_llm(form_files):
    form_names = [os.path.splitext(f)[0].replace('_', ' ') for f in form_files]

    # Step 1: Present options to user
    intro_conversation = [{
        "role": "system",
        "content": (
            "You are a friendly assistant. Greet the user and show them the following forms available to fill:\n"
            + "\n".join(f"- {name}" for name in form_names) +
            "\nThen ask: 'Which form would you like to fill?'"
        )
    }]

    intro_response = chat_completion(intro_conversation)
    print("🤖 Assistant:", parse_response(intro_response))

    # Step 2: Get user's selection and map it via LLM to file
    filename_conversation = [{
        "role": "system",
        "content": (
            "The following form files are available:\n" +
            "\n".join(form_files) +
            "\nGiven the user's response, return the best matching file name from the list above.\n"
            "Respond ONLY with the exact file name, no explanation."
        )
    }]

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            return None

        filename_conversation.append({"role": "user", "content": user_input})
        match_response = chat_completion(filename_conversation)
        match = parse_response(match_response).strip()

        if match in form_files:
            return match
        else:
            print("❌ Could not match your response to any form. Please try again.")
            filename_conversation.pop()


# ---------------- MAIN INTERACTION ----------------
def process_user_message(user_input, conversation, answers, answer_file_path):
    exit_phrases = ["i'm done", "done answering", "no more questions", "bye", "that's all"]
    conversation.append({"role": "user", "content": user_input})

    # ✅ Handle exit
    if any(p in user_input.lower() for p in exit_phrases):
        save_json(answer_file_path, answers)
        return "✅ Conversation ended. Thank you for your responses!", conversation, answers, True

    # ✅ Identify unanswered fields
    unanswered = [k for k, v in answers.items() if v.get("Value", "").strip() == ""]
    matched_key = ask_llama_to_match_field(conversation, user_input, unanswered, answers)

    # ✅ Handle valid field match
    if matched_key:
        field_info = answers[matched_key]
        cleaned = extract_clean_answer(conversation, matched_key, field_info, user_input, answers)

        if cleaned:
            field_info["Value"] = cleaned.split("\n")[0].strip()


            # 🧠 Ask next question naturally
            next_unanswered = [k for k, v in answers.items() if v.get("Value", "").strip() == ""]
            if not next_unanswered:
                save_json(answer_file_path, answers)
                return "✅ All questions are answered and saved. Thank you!", conversation, answers, True

            followup_prompt = (
                f"Cool! Now let's smoothly continue. Out of the remaining fields: {', '.join(next_unanswered)}, "
                f"what’s a fun or casual way to ask about the next one — without repeating the last response? "
                f"Keep it witty, friendly, maybe throw in an emoji or two if it fits!"
            )

            conversation.append({"role": "user", "content": followup_prompt})
            followup_response = chat_completion(conversation, answers)
            followup_text = parse_response(followup_response).strip()
            conversation.append({"role": "assistant", "content": followup_text})

            return followup_text, conversation, answers, False
        else:
            # ❌ Retry with explanation
            retry_prompt = generate_retry_prompt(matched_key, field_info, user_input)
            conversation.append({"role": "user", "content": retry_prompt})
            retry_response = chat_completion(conversation, answers)
            retry_message = parse_response(retry_response).strip()
            conversation.append({"role": "assistant", "content": retry_message})
            return retry_message, conversation, answers, False

    # 🧠 If no match and all fields are empty — assume it's a greeting or vague input
    if all(v.get("Value", "").strip() == "" for v in answers.values()):
        first_field = list(answers.keys())[0]
        field_info = answers[first_field]

        force_prompt = (
            f"The user said: '{user_input}', which doesn't map clearly to a question.\n"
            f"Greet the user and kindly start the form by asking about the field: '{first_field}' "
            f"(type: {field_info['Type']}). Be friendly and natural."
        )
        conversation.append({"role": "user", "content": force_prompt})
        response = chat_completion(conversation, answers)
        assistant_message = parse_response(response).strip()
        conversation.append({"role": "assistant", "content": assistant_message})
        return assistant_message, conversation, answers, False

    # 🤷‍♀️ Otherwise continue the conversation generically
    response = chat_completion(conversation, answers)
    assistant_message = parse_response(response).strip()
    conversation.append({"role": "assistant", "content": assistant_message})

    # ✅ Final completion check
    unanswered_all = [k for k, v in answers.items() if v.get("Value", "").strip() == ""]
    complete = not unanswered_all
    if complete:
        save_json(answer_file_path, answers)
        return "✅ All questions are answered and saved. Thank you!", conversation, answers, True

    return assistant_message, conversation, answers, False

# ---------------- MAIN ----------------

def conversation_loop():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    form_files = [f for f in os.listdir(FORM_FOLDER) if f.endswith(".json")]

    selected_file = None
    while not selected_file:
        selected_file = select_form_via_llm(form_files)

    form_name = os.path.splitext(selected_file)[0]
    questions = load_json(os.path.join(FORM_FOLDER, selected_file))
    answers = {k: v for k, v in questions.items()}
    conversation = [{"role": "system", "content": build_system_prompt(questions)}]

    # ✅ Define out_file BEFORE the loop
    out_file = os.path.join(OUTPUT_FOLDER, f"{form_name}_answers.json")

    print(f"\n🟢 Starting interview for: {selected_file}\n(Press Enter on empty line to stop)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "":
            print("\n🛑 Conversation ended by user.")
            break

        msg, conversation, answers, done = process_user_message(user_input, conversation, answers, out_file)
        print(f"\nAssistant: {msg}")
        if done:
            break

    save_json(out_file, answers)
    save_json(CHAT_LOG_FILE, conversation)
    print(f"\n💾 Answers saved to {out_file}")
    print(f"💬 Chat log saved to {CHAT_LOG_FILE}")

def build_system_prompt(questions):
    fields = ", ".join(questions.keys())
    return (
        "You are a warm, funny, and human-like assistant helping someone fill out a form — "
        "but without making it feel like a boring form.\n"
        f"The user will provide answers related to the following topics: {fields}.\n"
        "Don't be robotic. Respond like a friend having a casual chat while collecting these details.\n"
        "Be empathetic, use emojis, jokes, or encouragement as appropriate. No need to repeat the user's answers.\n"
        "Your goal is to make them smile and finish the form smoothly."
    )


if __name__ == "__main__":
    conversation_loop()
