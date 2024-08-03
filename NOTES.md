# Dataset

- If the groundtruth of the dataset has multiple options, we are going to randomly pick one for each epoch.
- We will prepare two type of datasets. 1. OCR output in json format 2. Q&A with images
- We will use LLAVA 1.6 tempelate 
```
prompt = f"[INST] <image>\n{QUERY} [\INST] {GROUND_TRUTH}"
```
- You can use various languages in your dataset (I will use English for simplicity)

