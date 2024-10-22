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
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

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
        (len(train_loader.dataset) // (hparams.train_batch_size))
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

def step(model, batch, args):
    lm_labels = batch["target_ids"].to(args.evice)
    lm_labels[lm_labels[:, :] == model.tokenizer.pad_token_id] = -100

    outputs = model.forward(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        lm_labels=lm_labels,
        decoder_attention_mask=batch['target_mask']
    )

    if args.use_modified_network:
        output_hidden_state = outputs.decoder_hidden_states[-1] * (model.t5_model.model_dim ** -0.5)
        lm_logits_gen = model.t5_model.lm_head(output_hidden_state)

        bs, qlen, dim = output_hidden_state.size()

        def shape(x):
            return x.view(bs, -1, model.t5_model.config.num_heads, model.t5_model.config.d_kv).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(bs, -1, model.inner_dim)

        input_hidden_state = model.t5_model.get_encoder()(batch["source_ids"].to(args.device))[0]
        q = shape(model.q(output_hidden_state))
        v = shape(model.v(input_hidden_state))
        k = shape(model.k(input_hidden_state))

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = nn.Dropout(p=model.t5_model.config.dropout_rate)(weights)

        context = torch.matmul(weights, v)
        context = unshape(context)
        context = model.o(context)
        context = context * (model.t5_model.model_dim ** -0.5)

        lm_logits_ext = model.t5_model.lm_head(context)

        gate_layer = model.ff_gate(torch.cat((model.layer_norm_gen(output_hidden_state), model.layer_norm_ext(context)), dim=2))
        gate_layer_output = torch.sigmoid(gate_layer)

        lm_logits = (1 - gate_layer_output) * lm_logits_gen + gate_layer_output * lm_logits_ext

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
        return loss, lm_logits
    else:
        return outputs