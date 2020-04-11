import logging
import os
from typing import List

import torch
from torch.utils.data.dataset import Dataset

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ...training_args import DataProcessingArguments
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures


logger = logging.getLogger(__name__)


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    output_mode: str

    def __init__(self, args: DataProcessingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
        if local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate else "train",
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                str(args.task_name),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            self.features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                RobertaTokenizer,
                RobertaTokenizerFast,
                XLMRobertaTokenizer,
            ):
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            examples = (
                processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
            )
            self.features = glue_convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                label_list=label_list,
                output_mode=self.output_mode,
            )
            if local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

        if local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
