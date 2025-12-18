from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""

    if request.method == "POST":
        prompt = request.form["prompt"]
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        outputs = model.generate(
            inputs,
            max_length=80,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template("index.html", output=generated_text)

if __name__ == "__main__":
    app.run(debug=True)
