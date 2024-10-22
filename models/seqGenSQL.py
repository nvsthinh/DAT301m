import argparse
from torch import nn

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
        self.t5_model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path).to(self.hparams.device)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path, legacy=False)

        if hparams.use_modified_network:
            self.inner_dim = self.t5_model.config.num_heads * self.t5_model.config.d_kv
            self.q = nn.Linear(self.t5_model.config.d_model, self.inner_dim, bias=False).to(self.hparams.device)
            self.k = nn.Linear(self.t5_model.config.d_model, self.inner_dim, bias=False).to(self.hparams.device)
            self.v = nn.Linear(self.t5_model.config.d_model, self.inner_dim, bias=False).to(self.hparams.device)
            self.layer_norm_gen = LayerNorm(self.t5_model.config.d_model, eps=self.t5_model.config.layer_norm_epsilon).to(self.hparams.device)
            self.layer_norm_ext = LayerNorm(self.t5_model.config.d_model, eps=self.t5_model.config.layer_norm_epsilon).to(self.hparams.device)
            self.ff_gate = nn.Linear(self.t5_model.config.d_model * 2, 1, bias=False).to(self.hparams.device)
            self.o = nn.Linear(self.inner_dim, self.t5_model.config.d_model, bias=False).to(self.hparams.device)

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

        outputs = self.t5_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
            output_hidden_states=True
        )
        return outputs