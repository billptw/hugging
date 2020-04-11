import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from .data.data_collator import DataCollator
from .modeling_utils import PreTrainedModel
from .training_args import TrainingArguments


try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


logger = logging.getLogger(__name__)


class Trainer:
    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    tb_writer: Optional["SummaryWriter"] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if is_tensorboard_available():
            self.tb_writer = SummaryWriter()

    def set_seed():
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    
    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator.collate_batch,
        )

    def train(self):
        train_dataloader = self.get_train_dataloader()
        train_iterator = trange(0, int(self.args.num_train_epochs), desc="Epoch")
        global_step = 0
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                loss = self._training_step(batch)
                global_step += 1
                print(global_step, loss)

    def _training_step(self, batch) -> float:
        self.model.train()
        inputs = self.data_processor.preprocess_batch(batch)
        outputs = self.model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        return loss.item()

    def is_master(self) -> bool:
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_if_master(self):
        """
        Saving best-practices: if you use defaults names for the model,
        you can reload it using from_pretrained().
        """
        if self.is_master():
            self._save()

    def _save(self):
        # Create output directory if needed
        if self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir, exists_ok=True)

        logger.info("Saving model checkpoint to %s", self.args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))
