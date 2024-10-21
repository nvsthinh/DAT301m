from transformers import (
    AdamW,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from dataset import WikiSqlDataset
from torch.utils.data import DataLoader
import random
import numpy as np
import torch

def get_dataset(hparams, data_type):
    return WikiSqlDataset(
        tokenizer=T5Tokenizer.from_pretrained(hparams.model_name_or_path),
        data_dir=hparams.data_dir,
        dataset_type=data_type,
        include_data_type=hparams.include_data_type,
        include_sample_data=hparams.num_sample_rows,
        data_augmentation=hparams.data_aug,
        generated_data=hparams.generated_data_files,
        max_input_len=hparams.max_seq_length,
        max_output_len=hparams.max_output_length
    )

def train_dataloader(hparams):
    train_dataset = get_dataset(hparams, data_type="train")
    return DataLoader(train_dataset, batch_size=hparams.train_batch_size, shuffle=True, num_workers=hparams.num_of_workers)

def val_dataloader(hparams):
    val_dataset = get_dataset(hparams, data_type="dev")
    return DataLoader(val_dataset, batch_size=hparams.eval_batch_size, num_workers=hparams.num_of_workers)

def configure_optimizers(hparams, model, train_loader):
    t_total = (
        (len(train_loader.dataset) // (hparams.train_batch_size * max(1, hparams.n_gpu)))
        // hparams.gradient_accumulation_steps
        * float(hparams.num_train_epochs)
    )

    optimizer = AdamW(model.parameters(), lr=hparams.learning_rate, eps=hparams.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=hparams.warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)