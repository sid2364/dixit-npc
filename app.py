#!/usr/bin/env python3
import os
import random
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, session
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

import os
random_key = os.urandom(24)

PROMPT = "Generate a surreal, dream-like, and poetic caption in the style of Dixit for the image. Give me one just sentence or phrase: "

# ------------------------------
# Set up Flask app and config
# ------------------------------
app = Flask(__name__)
app.secret_key = random_key # if we were in production, we would use an actual secret key
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models once at startup
# Use AutoProcessor instead of BlipProcessor
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

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


def creative_caption_direct(image, creativity_prompt=PROMPT):
    # Prepare inputs with the creative prompt.
    inputs = processor(image, text=creativity_prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_length=100,
        min_length=10,
        do_sample=True,
        top_p=0.9,
        temperature=1.2,
        num_return_sequences=1
    )
    print("output", output)
    caption = processor.decode(output[0], skip_special_tokens=True)
    print("caption", caption)
    return caption

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
    return chosen_index, basic_caption.split(PROMPT)[1]

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
      <label>Who is the storyteller?</label><br>
      <input type="radio" name="storyteller" value="computer" checked> Computer<br>
      <input type="radio" name="storyteller" value="player"> Player TODO<br><br>
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
                return redirect(url_for("player_storyteller"))
    return render_template_string(html)

# Computer as storyteller: display caption and let human guess
@app.route("/computer_storyteller", methods=["GET", "POST"])
def computer_storyteller():
    if request.method == "GET":
        chosen_index, creative_caption = computer_storyteller_turn(session['card_files'])
        session['chosen_index'] = chosen_index
        session['creative_caption'] = creative_caption

        # Build HTML to display the cards and the computer-generated caption
        html = """
        <h2>Computer as Storyteller</h2>
        <p><strong>Computer's Creative Caption:</strong> {{ caption }}</p>
        <p>Click the radio button of the card you think matches the caption:</p>
        <form method="post">
          {% for idx in range(card_files|length) %}
            <input type="radio" name="guess" value="{{ idx }}" required>
            <img src="{{ url_for('uploaded_file', filename=card_files[idx]) }}" width="150" style="margin:10px;">
          {% endfor %}
          <br><br>
          <input type="submit" value="Submit Guess">
        </form>
        """
        return render_template_string(html, caption=creative_caption, card_files=session['card_files'])
    else:
        # Process guess from human guesser
        guess = int(request.form.get('guess'))
        chosen_index = session.get('chosen_index')
        result = "Correct!" if guess == chosen_index else f"Wrong. The computer's card was #{chosen_index+1}."
        html = """
        <h2>Result</h2>
        <p>{{ result }}</p>
        <a href="{{ url_for('index') }}">Play Again</a>
        """
        return render_template_string(html, result=result)

if __name__ == "__main__":
    app.run(debug=True)