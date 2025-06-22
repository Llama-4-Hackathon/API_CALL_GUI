# ===== Python Libraries
import io
import os
import json
import base64
# import shutil

# ===== Imported Libraries
import requests
from PIL import Image
from tqdm import tqdm

# ===== PDF Libraries
from PyPDF2 import PdfReader
import pdfplumber
from pdf2image import convert_from_path

import GUI

SETTINGS_PATH = './config.json'
INPUT_FORM_DIR = './Input Forms'
OUTPUT_FORM_DIR = './Forms'


# Check if PDF is a Fillable PDF
def is_fillable_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    fields = reader.get_fields()

    # Return true if there are fields
    if fields:
        return True
    else:
        return False

# Check if PDF is Flat or Scanned PDF
def is_text_based_pdf(pdf_path):
    # Open PDF file
    with pdfplumber.open(pdf_path) as pdf:
        # Extract the text from all of the pages
        for page in pdf.pages:
            # Return true if page contains text
            if page.extract_text():
                return True


    # Return false if there is no text in the PDF file
    return False

# Return the type of PDF
def detect_pdf_type(pdf_path):
    # Check if the PFD is Fillable
    if is_fillable_pdf(pdf_path):
        return "fillable"

    # Check if the PDF is Flat
    elif is_text_based_pdf(pdf_path):
        return "text-based"

    # The PDF is Scanned
    else:
        return "scanned"

# Convert a PDF to a base 64 image
def pdf_to_base64(pdf_path):
    # Convert a PDF into images
    images = convert_from_path(pdf_path)

    # Combine images into one image
    if len(images) > 1:
        # Get the widths and heights for all of the images
        widths, heights = zip(*(img.size for img in images))

        # Get the total height for the new image
        total_height = sum(heights)
        # Get the max width for the new image
        max_width = max(widths)

        # Create a new image with the total height and max width
        combined = Image.new("RGB", (max_width, total_height))

        # Offset for each image from origin
        y_offset = 0

        # Stack images on top of each other
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height

    # Else there is no need to combine images so set combined to the first image
    else:
        combined = images[0]

    # IO buffer for base 64 image
    buffered = io.BytesIO()

    # Save the combined image to the IO buffer
    combined.save(buffered, format="PNG")

    # Get the base 64 image
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_base64

# Get the JSON format of a PDF file
def get_json_from_pdf(pdf_path, settings):
    # Get the Llama API settings
    llama_settings = settings.get("llama", {})

    # Get the Llama API key and model to use
    api_key = llama_settings.get("api-key")
    model = llama_settings.get("model")

    # Check if the PDF is a scanned or text-based PDF
    if detect_pdf_type(pdf_path) in ["scanned", "text-based", "fillable"]:
        # Convert the PDF to a base 64 image
        base64_img = pdf_to_base64(pdf_path)

        # System prompt
        system_prompt = """You are a form analysis assistant. Given an image of a PDF form, your job is to extract all visually fillable fields and return their structure in valid raw JSON format only. Do not include markdown, code blocks, or any extra explanation.

                Use the following format:

                {
                    "Field Name": {
                        "Value": "",
                        "Type": "",
                        "Required": true,
                        "Options": [],
                        "Description": ""
                    }
                }

                Field Types:
                - text - free-form input
                - number - numeric values
                - boolean - yes/no or single checkboxes (Always include Options: ["Yes", "No"])
                - choice - a single option from a group (e.g. radio buttons, grouped selection)
                - multi-choice - checkboxes with multiple possible selections
                - date, time, datetime - date/time pickers
                - file, image, signature, location, email, phone, url - appropriate field types

                General Rules:
                - Only extract fields that are visually fillable - such as input boxes, checkboxes, radio buttons, or clearly defined response areas.
                - Do NOT extract instructional text, disclaimers, examples, or paragraphs phrased as questions unless they are attached to an actual checkbox or input field.
                - Do NOT include entire instructions or static form labels as fields. Only include data the user is expected to provide directly.
                - If a field appears as a set of related options (e.g. filing status), group them as a single "choice" field with all options listed in "Options".
                - If multiple subfields are labeled by line numbers (e.g. Line 8a through 8z), treat each subline as its own distinct field with a descriptive name - do NOT group them into a single field.
                - All boolean fields must include the standard options: ["Yes", "No"]

                Descriptions:
                - Every field must include a "Description" - a plain-English explanation of what the field is asking for.
                - Descriptions must be clear and context-independent - do NOT include references like "line 8" or "see line 25". Instead, summarize what those lines represent.
                - Scale the description non-linearly with complexity:
                - For simple fields (e.g. name, date), use brief instructions like "Enter your first name."
                - For medium-complexity fields, use 5-10 explanatory sentences.
                - For complex calculations, credits, or deductions, generate long, detailed descriptions - even a full paragraph if needed - to explain what the user should input.

                Only respond with valid JSON. Do not include ```json or ``` at the beginning or end.
                """

        # User prompt and image
        user_prompt = "Analyze this PDF form and extract all fields as JSON"
        image_url = f"data:image/jpeg;base64,{base64_img}"

        # Send the post request to the Llama API to extract the fields as JSON
        response = requests.post(
            url="https://api.llama.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    },
                ],
            },
        )

        # Get the JSON response from the Llama API
        json_response = response.json()
        model_output = json_response.get("completion_message", {})
        content_response = model_output.get("content", {})
        content = content_response.get("text")

        # Parse the output into the JSON extracted from the PDF
        content = str(content)
        content = content.replace("\\n", "").replace("\\", "")
        try:
            content = json.loads(content)
        except:
            print(f"An error occurred:\n{content}")
            exit()

        return content

    # Return None if the PDF is not a scanned or text-based PDF
    return None


def main():
    # Load settings if they're available
    try:
        with open(SETTINGS_PATH, "r") as file:
            settings = json.load(file)
    except:
        print("No settings found, please set your llama api key")
        return -1

    # Check if the JSON directory exists
    if not os.path.exists(OUTPUT_FORM_DIR):
        os.makedirs(OUTPUT_FORM_DIR)

    # Generate example forms
    # gen_fillable() # Generate a computer generated form with fillable input
    # gen_flat() # Generate a computer generated form without fillable input
    # gen_scanned() # Generate a scanned form

    for file in tqdm(os.listdir(INPUT_FORM_DIR)):
        # Get only pdf files from the form dir
        if os.path.splitext(file)[1] == ".pdf":
            # The path to the output dir of this file
            # output_path = os.path.join(OUTPUT_FORM_DIR, os.path.splitext(file)[0])

            # Path of input PDF
            pdf_path = os.path.join(INPUT_FORM_DIR, file)

            # Path of output JSON
            json_path = os.path.join(OUTPUT_FORM_DIR, f"{os.path.splitext(file)[0]}.json")

            # Get JSON data from PDF
            json_data = get_json_from_pdf(pdf_path, settings)

            # Make sure that we get JSON output
            if json_data is not None:
                # Write JSON data to output file
                with open(json_path, "w") as output_file:
                    output_file.write(json.dumps(json_data, indent=4))

    GUI.main()

if __name__ == "__main__":
    main()
