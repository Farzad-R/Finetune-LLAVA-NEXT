# Fine-Tune LLAVA Repository

This repository demonstrates the process of fine-tuning LLAVA for various tasks, including data parsing and extracting JSON information from images. It provides comprehensive guidance on how to handle different datasets and fine-tune the model effectively.

---

## Video Explanation: 
A detailed explanation of the project is available in the following YouTube video:

Fine-Tuning Multimodal LLMs (LLAVA) for Image Data Parsing: [Link](https://youtu.be/0pd1ZDT--mU?si=IvVdfgv5CXZx57Dr)

---

## Repository Structure

### Notebooks
- **`data_exploration/`**  
  Contains notebooks for exploring the [Cord-V2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) and [DocVQA](https://huggingface.co/datasets/nielsr/docvqa_1200_examples) datasets.

- **`fine-tuning/`**  
  Includes:
  - A notebook for fine-tuning LLAVA 1.6 7B
  - A notebook for testing the fine-tuned model

- **`test_model/`**  
  Contains multiple notebooks for testing:
  - LLAVA 1.5 7B and 13B
  - LLAVA 1.6 7B, 13B, and 34B

### Source Code
- **`src/`**  
  Contains a Streamlit app to showcase the performance of the fine-tuned model. 

  To run the dashboard:
  1. In Terminal 1:
     ```bash
     python src/serve_model.py
     ```
  2. In Terminal 2:
     ```bash
     streamlit run src/app.py
     ```
  Open the dashboard at [http://localhost:8501/](http://localhost:8501/) and upload sample images from the `data` folder to view the results. You can find 20 sample images in the `data` folder.

---

## Installation

1. Clone this repository using:
    ```bash
    git clone https://github.com/Farzad-R/Finetune-LLAVA-NEXT.git
    ```

2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional requirements:
   ```bash
   pip install git+https://github.com/huggingface/transformers.git
   ```
---

## Additional Resources

- [Link to Hyperstack Cloud](https://www.hyperstack.cloud/?utm_source=Influencer&utm_medium=AI%20Round%20Table&utm_campaign=Video%201)
- [HuggingFace Hub to access the model](https://huggingface.co/Farzad-R/llava-v1.6-mistral-7b-cordv2)
- A link to a YouTube video will be added here soon to provide further insights and demonstrations.
- LLAVA-NEXT [models](https://huggingface.co/docs/transformers/en/model_doc/llava_next).
- LLAVA-NEXT [info](https://llava-vl.github.io/blog/2024-01-30-llava-next/).
- LLAVA-NEXT [demo](https://huggingface.co/spaces/lmms-lab/LLaVA-NeXT-Interleave-Demo).
- LLAVA-NEXT GitHub [repository](https://github.com/LLaVA-VL/LLaVA-NeXT).
- LLAVA 1.5 [demo](https://llava.hliu.cc/).
- LLAVA 1.5 GitHub [repository](https://github.com/haotian-liu/LLaVA).


---