from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Any, Dict
import random
import json


class LlavaDataset(Dataset):
    """
    PyTorch Dataset for LLaVa, designed to work with datasets from HuggingFace.

    This class takes a HuggingFace Dataset as input and processes each entry,
    which consists of an image path (in png/jpg/jpeg format) and corresponding
    ground truth data (in json/jsonl/txt format).

    Attributes:
        split (str): The dataset split to load (e.g., "train", "test").
        sort_json_key (bool): If True, keys in the JSON ground truth are sorted.
        dataset (Dataset): The HuggingFace dataset loaded according to the specified split.
        dataset_length (int): The number of samples in the dataset.
        gt_token_sequences (List[List[str]]): A list of tokenized ground truth sequences
            for each sample in the dataset.

    Args:
        dataset_name_or_path (str): The name of the dataset or the path to the dataset files.
        split (str): The dataset split to use. Default is "train".
        sort_json_key (bool): Whether to sort keys in JSON ground truth. Default is True.
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert a JSON object into a token sequence string.

        Args:
            obj (Any): The JSON object to convert, which can be a dictionary, list, or other types.
            sort_json_key (bool): Whether to sort the keys of a dictionary. Default is True.

        Returns:
            str: A string representing the token sequence extracted from the JSON object.
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieve a single data point from the dataset by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple:
                - image: The image associated with the sample.
                - target_sequence: A string representing the tokenized ground truth sequence.
        """
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1

        return image, target_sequence
    

def train_collate_fn(examples, processor, max_length:int):
    """
    Prepares a batch of data for training by processing images and corresponding 
    ground truth data into a format suitable for input into a model.

    Args:
        examples (List[Tuple[Any, str]]): A list of examples, where each example is a tuple 
            consisting of an image and its corresponding ground truth data (as a string).
        processor (PreTrainedProcessor): A processor object used to prepare the images and text.
        max_length (int): The maximum length for the tokenized input sequences.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
            - input_ids (torch.Tensor): Token IDs of the input sequences.
            - attention_mask (torch.Tensor): Attention mask indicating non-padding tokens.
            - pixel_values (torch.Tensor): Preprocessed images ready for model input.
            - image_sizes (List[Tuple[int, int]]): Original sizes of the images.
            - labels (torch.Tensor): Labels for the training data, with padding tokens masked as -100.
    """

    images = []
    texts = []
    for example in examples:
        image, ground_truth = example
        images.append(image)
        # TODO: in the future we can replace this by processor.apply_chat_template
        prompt = f"[INST] <image>\nExtract JSON [\INST] {ground_truth}"
        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, image_sizes, labels


def eval_collate_fn(examples, processor):
    """
    Prepares a batch of data for evaluation by processing images and corresponding 
    ground truth data into a format suitable for input into a model. The ground truth 
    data is separated for later comparison with model predictions.

    Args:
        examples (List[Tuple[Any, str]]): A list of examples, where each example is a tuple 
            consisting of an image and its corresponding ground truth data (as a string).
        processor (PreTrainedProcessor): A processor object used to prepare the images and text.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[str]]:
            - input_ids (torch.Tensor): Token IDs of the input sequences.
            - attention_mask (torch.Tensor): Attention mask indicating non-padding tokens.
            - pixel_values (torch.Tensor): Preprocessed images ready for model input.
            - image_sizes (List[Tuple[int, int]]): Original sizes of the images.
            - answers (List[str]): The ground truth answers for each example.
    """
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        image, ground_truth = example
        images.append(image)
        # TODO: in the future we can replace this by processor.apply_chat_template
        prompt = f"[INST] <image>\nExtract JSON [\INST]"
        texts.append(prompt)
        answers.append(ground_truth)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers