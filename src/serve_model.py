from flask import Flask, request, jsonify
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import io
import base64
import re

app = Flask(__name__)

# Initialize the model and processor
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
REPO_ID = "Farzad-R/llava-v1.6-mistral-7b-cordv2"

processor = AutoProcessor.from_pretrained(MODEL_ID)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)

# Function to convert tokens to JSON
def token2json(tokens, is_inner_value=False, added_vocab=None):
    if added_vocab is None:
        added_vocab = processor.tokenizer.get_added_vocab()

    output = {}

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        key_escaped = re.escape(key)

        end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
            )
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_base64 = data['image']

    # Decode base64 to bytes
    image_bytes = base64.b64decode(image_base64)

    # Load image
    image = Image.open(io.BytesIO(image_bytes))

    # Prepare input
    prompt = f"[INST] <image>\nExtract JSON [/INST]"
    max_output_token = 256
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=max_output_token)
    response = processor.decode(output[0], skip_special_tokens=True)

    # Convert response to JSON
    generated_json = token2json(response)

    return jsonify(generated_json)

if __name__ == '__main__':
    app.run(port=5000, debug=False)
