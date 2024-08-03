from torch.utils.data import Dataset
from typing import Any, Dict
import random
from datasets import load_dataset

class LlavaDataset(Dataset):
    """
    PyTorch Dataset for LLaVa. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
    ):
        super().__init__()

        self.split = split

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset_length = len(self.dataset)

        self.answer_token_sequences = []
        self.query_list = []
        for sample in self.dataset:
            if "answers" in sample:
                assert isinstance(sample["answers"], list)
                self.answer_token_sequences.append(sample["answers"])
                self.query_list.append(sample["query"]["en"])


    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # inputs
        image = sample["image"]
        en_query = self.query_list[idx]
        target_sequence = random.choice(self.answer_token_sequences[idx]) # can be more than one, e.g., DocVQA Task 1
        return image, en_query, target_sequence
    
