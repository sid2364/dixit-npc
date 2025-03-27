#!/usr/bin/env python3
import os
import random
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, session
from PIL import Image
import torch
# from transformers import AutoProcessor, Blip2ForConditionalGeneration
# from transformers import ViltProcessor, ViltForMaskedLM

random_key = os.urandom(24)
prompt = "Generate a surreal, dream-like, and poetic caption in the style of Dixit. It shouldn't be too literal, but should evoke a sense of wonder and mystery."

# ------------------------------
# Set up Flask app and config
# ------------------------------
app = Flask(__name__)
app.secret_key = random_key # if we were in production, we would use an actual secret key, but I'm never planning to productionize this
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models once at startup
# Use AutoProcessor instead of BlipProcessor
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load model (do this once at startup)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map="auto",  # Let it smartly split across CPU/GPU
    torch_dtype=torch.float16  # Optional: reduces memory usage
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def creative_caption_direct_old(image):
    # Process the image and generate caption
    inputs = processor(image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=160,
            do_sample=True,
            top_p=0.9,
            temperature=1.3,
            num_return_sequences=1
        )

    # Decode the output
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    print("outputs", processor.decode(outputs[0], skip_special_tokens=True))
    # Remove the prompt from the caption
    caption = caption.replace(prompt, "").strip()

    return caption


def two_step_caption_generation(image):
    """
    Generate a caption using a two-step reasoning process:
    1. First get a detailed description of the image
    2. Feed that to the model and ask it to create a creative caption
    """
    # Step 1
    description_prompt = (
        """
        Provide an extremely detailed, comprehensive, and nuanced description of this image. Describe every visual element, color, texture, mood, and potential symbolic meaning. Include minute details, spatial relationships, and any subtle or implied narratives. Your description should be rich, elaborate, and capture both the literal and metaphorical aspects of the image."
        """
    )

    # Process the image for detailed description
    inputs = processor(image, text=description_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        detailed_description_outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.9,
            temperature=1.1,
            num_return_sequences=1
        )

    # Decode the detailed description
    detailed_description = processor.decode(detailed_description_outputs[0], skip_special_tokens=True)
    print("detailed_description", detailed_description)
    detailed_description = detailed_description.replace(description_prompt, "").strip()

    print("stripped detailed_description", detailed_description)

    # Step 2
    caption_prompt = (
        """Based on this detailed image description, create a surreal, poetic, and imaginative Dixit-style caption. The caption should capture the essence and underlying emotions of the image, not its literal content. Draw metaphorical and abstract connections. Avoid direct references to specific objects. Description to inspire caption: """ + detailed_description
    )

    print("caption_prompt", caption_prompt)

    # Process for caption generation
    caption_inputs = processor(image, text=caption_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        caption_outputs = model.generate(
            **caption_inputs,
            max_new_tokens=80,
            do_sample=True,
            top_p=0.9,
            temperature=1.3,
            num_return_sequences=3,
            no_repeat_ngram_size=2
        )

    # Decode and process captions
    captions = [
        processor.decode(output, skip_special_tokens=True).replace(caption_prompt, "").strip()
        for output in caption_outputs
    ]

    # Select the most unique and creative caption
    final_caption = max(set(captions), key=captions.count)

    return {
        'detailed_description': detailed_description,
        'caption': final_caption
    }


# Modify your existing function to use this new approach
def creative_caption_direct(image):
    result = two_step_caption_generation(image)
    return result['caption']

# Utility Funcs
def split_cards(image_path, num_rows=2, num_cols=3):
    """
    Split an image with cards arranged in a num_rows x num_cols grid into individqal card images
    """
    image = Image.open(image_path)
    width, height = image.size

    # Calculate the size of each card
    card_width = width // num_cols
    card_height = height // num_rows

    cards = []
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * card_width
            right = (col + 1) * card_width
            top = row * card_height
            bottom = (row + 1) * card_height

            # Crop out the card
            card = image.crop((left, top, right, bottom))
            cards.append(card)

    return cards

# TODO: After each round, delete the card images from the folder

def save_cards(cards, prefix="card"):
    """Save each card image to UPLOAD_FOLDER; return list of filenames"""
    filenames = []
    for i, card in enumerate(cards):
        filename = f"{prefix}_{random.randint(1000,9999)}_{i+1}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        card.save(path)
        filenames.append(filename)
    return filenames

def computer_storyteller_turn(cards):
    """Computer selects a random card, generates a basic and creative captio"""
    chosen_index = random.randint(0, len(cards)-1)
    # Load the chosen card from file
    chosen_path = os.path.join(UPLOAD_FOLDER, session['card_files'][chosen_index])
    card_image = Image.open(chosen_path)
    # basic_caption = generate_basic_caption(card_image)
    basic_caption = creative_caption_direct(card_image)
    # creative_caption = creative_rewrite(basic_caption)

    # Return only the caption after the prompt
    return chosen_index, basic_caption

# Flask stuff, routes and views

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Home page: upload image and choose storyteller
@app.route("/", methods=["GET", "POST"])
def index():
    html = """
    <h1>Dixit</h1>
    <form method="post" enctype="multipart/form-data">
      <label for="file">Upload an image with 6 cards (2x3 cards, 2 rows, 3 cards each):</label>
      <input type="file" name="file" required><br><br>
      <label>Whose hand is it?</label><br>
      <input type="radio" name="storyteller" value="computer" checked> Computer<br>
      <input type="radio" name="storyteller" value="player"> Me<br><br>
      <input type="submit" value="Upload and Start">
    </form>
    """
    if request.method == "POST":
        # Save uploaded file
        file = request.files['file']
        if file:
            filename = f"uploaded_{random.randint(1000,9999)}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['uploaded_file'] = filename

            # Split image into cards
            cards = split_cards(filepath)
            card_files = save_cards(cards)
            session['card_files'] = card_files  # store list in session

            # Store storyteller mode in session
            session['storyteller'] = request.form.get('storyteller', 'computer')

            # Redirect based on storyteller type
            if session['storyteller'] == "computer":
                return redirect(url_for("computer_storyteller"))
            else:
                return redirect(url_for("choose_card"))
    return render_template_string(html)


# New route: let the user choose which card to caption
@app.route("/choose_card", methods=["GET", "POST"])
def choose_card():
    if request.method == "GET":
        # Display all cards for selection
        html = """
        <h2>Select a card to generate its caption</h2>
        <form method="post">
          {% for idx in range(card_files|length) %}
            <input type="radio" name="selected_card" value="{{ idx }}" required>
            <img src="{{ url_for('uploaded_file', filename=card_files[idx]) }}" width="150" style="margin:10px;">
          {% endfor %}
          <br><br>
          <input type="submit" value="Generate Caption">
        </form>
        """
        return render_template_string(html, card_files=session['card_files'])
    else:
        # Process the selected card and generate its caption
        selected_index = int(request.form.get('selected_card'))
        selected_filename = session['card_files'][selected_index]
        chosen_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_filename)
        card_image = Image.open(chosen_path)
        caption = creative_caption_direct(card_image)
        html = """
        <h2>Caption for the selected card</h2>
        <img src="{{ url_for('uploaded_file', filename=selected_filename) }}" width="150" style="margin:10px;">
        <p><strong>Generated Caption:</strong> {{ caption }}</p>
        <a href="{{ url_for('index') }}">Back to Home</a>
        """
        return render_template_string(html, selected_filename=selected_filename, caption=caption)

# Computer as storyteller: display caption and let human guess
@app.route("/computer_storyteller", methods=["GET"])
def computer_storyteller():
    # Pick one card from the hand and generate its caption
    chosen_index, creative_caption = computer_storyteller_turn(session['card_files'])
    chosen_file = session['card_files'][chosen_index]

    # Build HTML to display the chosen card and its generated caption directly
    html = """
    <h2>Computer as Storyteller</h2>
    <p><strong>Computer's Creative Caption:</strong> {{ caption }}</p>
    <img src="{{ url_for('uploaded_file', filename=chosen_file) }}" width="150" style="margin:10px;">
    <br><br>
    <a href="{{ url_for('index') }}">Play Again</a>
    """
    return render_template_string(html, caption=creative_caption, chosen_file=chosen_file)

if __name__ == "__main__":
    app.run(debug=True)