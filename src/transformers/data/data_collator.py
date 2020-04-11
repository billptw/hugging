from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from ..tokenization_utils import PreTrainedTokenizer
from .processors.utils import InputFeatures


@dataclass
class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    def collate_batch(self):
        """
        Take a list of samples from a Dataset and collate them into a batch.
        """
        pass

    def preprocess_batch(self, batch) -> Dict[str, torch.Tensor]:
        """
        Take a batch, and return a dict of model inputs.

        Default implementation is identity.
        """
        return batch


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


@dataclass
class DataCollatorForSequenceClassification(DataCollator):

    output_mode: OutputMode = OutputMode.classification

    def collate_batch(self, features: List[InputFeatures]) -> Dict[str, torch.Tensor]:
        if self.output_mode == "classification":
            labels = torch.tensor([f.label for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label for f in features], dtype=torch.float)
        return {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            "token_type_ids": torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            "labels": labels,
        }


@dataclass
class DataCollatorForLM(DataCollator):
    tokenizer: Optional[PreTrainedTokenizer] = None
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def preprocess_batch(self, batch) -> Dict[str, torch.Tensor]:
        inputs, labels = mask_tokens(batch, self.tokenizer, self)  # noqa
        return {"input_ids": inputs, "masked_lm_labels": labels}
