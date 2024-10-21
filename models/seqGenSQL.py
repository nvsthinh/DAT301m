import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
    )
from models.layer_norm import LayerNorm

######################################################################
## T5 Model with modified layer for WikiSQL
######################################################################
class SeqGenSQL(nn.Module):
    def __init__(self, hparams):
        super(SeqGenSQL, self).__init__()

        if not isinstance(hparams, argparse.Namespace):
            hparams = argparse.Namespace(**hparams)

        self.hparams = hparams

        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)

        # Move model to the specified device
        self.model.to(self.hparams.device)

        if hparams.use_modified_network:
            self.inner_dim = self.model.config.num_heads * self.model.config.d_kv
            self.q = nn.Linear(self.model.config.d_model, self.inner_dim, bias=False)
            self.k = nn.Linear(self.model.config.d_model, self.inner_dim, bias=False)
            self.v = nn.Linear(self.model.config.d_model, self.inner_dim, bias=False)
            self.layer_norm_gen = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
            self.layer_norm_ext = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
            self.ff_gate = nn.Linear(self.model.config.d_model * 2, 1, bias=False)
            self.o = nn.Linear(self.inner_dim, self.model.config.d_model, bias=False)

        # Move additional layers to the specified device
        if hparams.use_modified_network:
            self.q.to(self.hparams.device)
            self.k.to(self.hparams.device)
            self.v.to(self.hparams.device)
            self.layer_norm_gen.to(self.hparams.device)
            self.layer_norm_ext.to(self.hparams.device)
            self.ff_gate.to(self.hparams.device)
            self.o.to(self.hparams.device)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        # Move inputs to device
        input_ids = input_ids.to(self.hparams.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.hparams.device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.hparams.device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.hparams.device)
        if lm_labels is not None:
            lm_labels = lm_labels.to(self.hparams.device)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
            output_hidden_states=True
        )
        return outputs

    def step(self, batch):
        lm_labels = batch["target_ids"].to(self.hparams.device)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        if self.hparams.use_modified_network:
            output_hidden_state = outputs.decoder_hidden_states[-1] * (self.model.model_dim ** -0.5)
            lm_logits_gen = self.model.lm_head(output_hidden_state)

            bs, qlen, dim = output_hidden_state.size()

            def shape(x):
                return x.view(bs, -1, self.model.config.num_heads, self.model.config.d_kv).transpose(1, 2)

            def unshape(x):
                return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

            input_hidden_state = self.model.get_encoder()(batch["source_ids"].to(self.hparams.device))[0]
            q = shape(self.q(output_hidden_state))
            v = shape(self.v(input_hidden_state))
            k = shape(self.k(input_hidden_state))

            scores = torch.einsum("bnqd,bnkd->bnqk", q, k)
            weights = F.softmax(scores.float(), dim=-1).type_as(scores)
            weights = nn.Dropout(p=self.model.config.dropout_rate)(weights)

            context = torch.matmul(weights, v)
            context = unshape(context)
            context = self.o(context)
            context = context * (self.model.model_dim ** -0.5)

            lm_logits_ext = self.model.lm_head(context)

            gate_layer = self.ff_gate(torch.cat((self.layer_norm_gen(output_hidden_state), self.layer_norm_ext(context)), dim=2))
            gate_layer_output = torch.sigmoid(gate_layer)

            lm_logits = (1 - gate_layer_output) * lm_logits_gen + gate_layer_output * lm_logits_ext

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss, lm_logits
        else:
            return outputs