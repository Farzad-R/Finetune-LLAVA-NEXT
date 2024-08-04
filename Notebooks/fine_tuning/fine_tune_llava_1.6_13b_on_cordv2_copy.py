from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoProcessor
import torch
from LlavaDataset import LlavaDataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from PushToHubCallback import PushToHubCallback
from utils import find_all_linear_names

MAX_LENGTH = 256
MODEL_ID = "llava-hf/llava-v1.6-vicuna-13b-hf"
REPO_ID = "Farzad-R/llava-v1.6-vicuna-13b-cordv2"
NUM_WORKERS = 4

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

train_dataset = LlavaDataset("naver-clova-ix/cord-v2",  split="train", sort_json_key=False)
val_dataset = LlavaDataset("naver-clova-ix/cord-v2", split="validation", sort_json_key=False)

counter = 0
for idx in range(len(train_dataset)):
    image, target_sequence = train_dataset[idx]
    # print(f"[INST] <image>\nExtract JSON [\INST] {target_sequence}")
    counter +=1
    if counter == 10:
        break

def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        image, ground_truth = example
        images.append(image)
        prompt = f"<s> <image>\nExtract JSON ASSISTANT </s> {ground_truth}"
        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    # input_ids = batch["input_ids"]
    # attention_mask = batch["attention_mask"]
    # pixel_values = batch["pixel_values"]
    # image_sizes = batch["image_sizes"]
    # labels = batch["labels"]

    # return input_ids, attention_mask, pixel_values, image_sizes, labels
    return batch


def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        image, ground_truth = example
        images.append(image)
        prompt = f"<s> <image>\nExtract JSON ASSISTANT </s>"
        texts.append(prompt)
        answers.append(ground_truth)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers


import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        # input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # pixel_values = pixel_values.to(self.device)
        # image_sizes = image_sizes.to(self.device)
        # labels = labels.to(self.device)

        # outputs = self.model(input_ids=input_ids,
        #                     attention_mask=attention_mask,
        #                     pixel_values=pixel_values,
        #                     image_sizes=image_sizes,
        #                     labels=labels
        #                   )
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        batch = {k: v.to(self.device) for k, v in batch.items()}

        # input_ids, attention_mask, pixel_values, image_sizes, answers = batch
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # pixel_values = pixel_values.to(self.device)
        # image_sizes = image_sizes.to(self.device)

        # autoregressively generate token IDs
        # generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
        #                                pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=MAX_LENGTH)
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_sizes=batch["image_sizes"],
            max_new_tokens=MAX_LENGTH
        )
        # turn them back into text, chopping of the prompt
        # predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        predictions = self.processor.batch_decode(generated_ids[:, batch["input_ids"].size(1):], skip_special_tokens=True)


        scores = []
        # for pred, answer in zip(predictions, answers):
        for pred, answer in zip(predictions, batch["labels"]):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    

config = {"max_epochs": 1,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          # "seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
          "num_workers": NUM_WORKERS
}

model_module = LlavaModelPLModule(config, processor, model)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from huggingface_hub import HfApi
api = HfApi()

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

trainer = L.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=2,
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        callbacks=[
            PushToHubCallback(),
            early_stop_callback],
)

trainer.fit(model_module)