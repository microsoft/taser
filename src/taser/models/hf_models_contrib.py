# Copyright (c) 2023 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import get_model_obj, load_states_from_checkpoint
from torch import Tensor as T
from torch import nn
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)
from transformers.file_utils import ModelOutput
from transformers.optimization import Adafactor, AdamW

from taser.models.biencoder_contrib import BiEncoderV2, MoEBiEncoder
from taser.models.moe_models import MoEBertModel
from taser.models.optimization import AdamWLayer, get_layer_lrs, get_layer_lrs_for_t5

logger = logging.getLogger(__name__)

model_mapping = {
    "bert": (BertConfig, BertTokenizer, BertModel),
    "Luyu/co": (BertConfig, BertTokenizer, BertModel),
    "facebook/contriever": (BertConfig, BertTokenizer, BertModel),
    "roberta": (RobertaConfig, RobertaTokenizer, RobertaModel),
    "google/electra": (ElectraConfig, ElectraTokenizer, ElectraModel),
    "t5": (T5Config, T5Tokenizer, T5EncoderModel),
}

moe_model_mapping = {
    "facebook/contriever": (BertConfig, BertTokenizer, (MoEBertModel, BertModel)),
    "bert": (BertConfig, BertTokenizer, (MoEBertModel, BertModel)),
    "Luyu/co": (BertConfig, BertTokenizer, (MoEBertModel, BertModel)),
}


def get_any_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    use_vat = False
    # TODO(chenghao): fix this.
    use_moe = cfg.encoder.use_moe
    num_expert = 0
    num_q_expert = None
    num_ctx_expert = None
    if use_moe:
        if not hasattr(cfg.encoder, "num_expert") or cfg.encoder.num_expert == -1:
            raise ValueError("When use_moe=True, num_expert is required")
        num_expert = cfg.encoder.num_expert
        use_infer_expert = cfg.encoder.use_infer_expert
        per_layer_gating = cfg.encoder.per_layer_gating
        moe_type = cfg.encoder.moe_type
        if hasattr(cfg.encoder, "num_q_expert"):
            num_q_expert = cfg.encoder.num_q_expert
            num_ctx_expert = num_expert - num_q_expert
        else:
            print("No num_q_expert is set")
    else:
        num_expert = 1
        use_infer_expert = False
        per_layer_gating = False
        moe_type = "mod3"

    if hasattr(cfg, "train") and hasattr(cfg.train, "use_vat"):
        use_vat = cfg.train.use_vat
    if use_vat:
        print("Adversarial biencoder DPR.")
    else:
        print("Vanilla biencoder DPR.")

    mean_pool_q_encoder = False
    if cfg.encoder.mean_pool:
        mean_pool_q_encoder = True
        if cfg.encoder.mean_pool_ctx_only:
            logger.info("Uses mean-pooling for context encoder only.")
            mean_pool_q_encoder = False

    factor_rep = cfg.encoder.factor_rep

    question_encoder = HFEncoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        use_vat=use_vat,
        use_moe=use_moe,
        moe_type=moe_type,
        use_infer_expert=use_infer_expert,
        per_layer_gating=per_layer_gating,
        num_expert=num_expert if cfg.encoder.shared_encoder else num_q_expert,
        mean_pool=mean_pool_q_encoder,
        factor_rep=factor_rep,
        pretrained=cfg.encoder.pretrained,
        **kwargs,
    )

    if cfg.encoder.shared_encoder:
        logger.info("Uses a shared encoder for both question and context.")
        ctx_encoder = question_encoder
    else:
        ctx_encoder = HFEncoder(
            cfg.encoder.pretrained_model_cfg,
            projection_dim=cfg.encoder.projection_dim,
            use_vat=use_vat,
            use_moe=use_moe,
            moe_type=moe_type,
            use_infer_expert=use_infer_expert,
            per_layer_gating=per_layer_gating,
            num_expert=num_ctx_expert if num_ctx_expert else num_expert,
            dropout=dropout,
            mean_pool=cfg.encoder.mean_pool,
            factor_rep=factor_rep,
            pretrained=cfg.encoder.pretrained,
            **kwargs,
        )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    if use_moe:
        logger.info("Using MOE model")
        if num_q_expert is not None:
            offet_expert_id = True
        else:
            offset_expert_id = False
        biencoder = MoEBiEncoder(
            question_encoder,
            ctx_encoder,
            fix_ctx_encoder=fix_ctx_encoder,
            num_expert=num_expert,
            num_q_expert=num_q_expert,
            offset_expert_id=offset_expert_id,
        )

    elif use_vat:
        raise NotImplementedError()
    else:
        biencoder = BiEncoderV2(
            question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
        )

    if cfg.encoder.pretrained_file:
        logger.info(
            "loading biencoder weights from %s, this should be a trained DPR model checkpoint",
            cfg.encoder.pretrained_file,
        )
        checkpoint = load_states_from_checkpoint(cfg.encoder.pretrained_file)
        model_to_load = get_model_obj(biencoder)
        model_to_load.load_state(checkpoint)

    if "base" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    elif "large" in cfg.encoder.pretrained_model_cfg:
        n_layers = 24
    elif "condenser" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    elif "contriever" in cfg.encoder.pretrained_model_cfg:
        n_layers = 12
    else:
        raise ValueError("Unknown nlayers for %s" % cfg.encoder.pretrained_model_cfg)

    if not inference_only and hasattr(cfg.train, "moe_factor"):
        moe_factor = cfg.train.moe_factor
    else:
        moe_factor = 1.0
    optimizer = (
        get_optimizer_v2(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            adam_betas=cfg.train.adam_betas,
            weight_decay=cfg.train.weight_decay,
            use_layer_lr=cfg.train.use_layer_lr,
            n_layers=n_layers,
            layer_decay=cfg.train.layer_decay,
            opt_name=cfg.train.opt_name if hasattr(cfg.train, "opt_name") else "adam",
            use_t5=("t5" in cfg.encoder.pretrained_model_cfg),
            moe_factor=moe_factor,
        )
        if not inference_only
        else None
    )

    tensorizer = get_any_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_any_tensorizer(cfg, tokenizer=None):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    if not tokenizer:
        tokenizer = get_any_tokenizer(
            pretrained_model_cfg, do_lower_case=cfg.do_lower_case
        )
        if cfg.special_tokens:
            from dpr.models.hf_models import _add_special_tokens

            _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizerV2(tokenizer, sequence_length)  # this should be fine


def get_any_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    model_name = pretrained_cfg_name.split("-")[0]
    tokenizer_class = model_mapping[model_name][1]
    logger.info("The tokenizer used is %s", tokenizer_class.__name__)
    return tokenizer_class.from_pretrained(pretrained_cfg_name)


class SimHeader(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, dropout: float = 0.1, eps: float = 1e-6
    ):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_(mean=0.0, std=0.02)
        self.project = nn.Sequential(linear, nn.LayerNorm(out_dim))
        self.out_features = out_dim

    def forward(self, inputs):
        return self.project(inputs)


import regex


def init_moe_from_pretrained_mapping(pretrained_sd, moe_sd, moe_layer_name="moe_layer"):
    state_dict = {}
    missing_vars = []
    pattern_list = [
        (f"{moe_layer_name}.", ""),
        (r"interm_layers.\d+", "intermediate"),
        (r"output_layers.\d+", "output"),
        (r"moe_query.\d+", "query"),
        (r"moe_key.\d+", "key"),
        (r"moe_value.\d+", "value"),
        (r"moe_dense.\d+", "dense"),
    ]

    def normalize_var_name(var_name):
        for ptn in pattern_list:
            var_name = regex.sub(ptn[0], ptn[1], var_name)
        return var_name

    for var_name in moe_sd:
        if moe_layer_name in var_name or "moe" in var_name:
            pretrained_var_name = normalize_var_name(var_name)
            logger.info(f"Loads {var_name} from {pretrained_var_name}")
        else:
            pretrained_var_name = var_name

        if pretrained_var_name in pretrained_sd:
            state_dict[var_name] = pretrained_sd[pretrained_var_name]
        else:
            missing_vars.append((var_name, pretrained_var_name))

    again_missing_vars = []
    for var_name, _ in missing_vars:
        if "expert_gate" in var_name:
            logger.info("Random init %s", var_name)
            state_dict[var_name] = moe_sd[var_name]
        else:
            again_missing_vars.append(var_name)

    if again_missing_vars:
        print("Missing again variables:", again_missing_vars)
    return state_dict


@dataclass
class HFEncoderOutput(ModelOutput):
    sequence_output: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    total_entropy_loss: torch.FloatTensor = None


class HFEncoder(nn.Module):
    def __init__(
        self,
        cfg_name: str,
        use_vat: bool = False,
        use_moe: bool = False,
        moe_type: str = "mod3",
        num_expert: int = 0,
        use_infer_expert: bool = False,
        per_layer_gating: bool = False,
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
        mean_pool: bool = False,
        factor_rep: bool = False,
        **kwargs,
    ):
        super().__init__()
        model_name = cfg_name.split("-")[0]
        if use_moe:
            logger.info("Using MoE models for HFEncoder")
            logger.info("Number of expert: %d", num_expert)
            config_class, _, model_class = moe_model_mapping[model_name]
            assert num_expert > 0, "num_expert can't be zero when using MoE."
        elif use_vat:
            raise NotImplementedError()
        else:
            config_class, _, model_class = model_mapping[model_name]
        cfg = config_class.from_pretrained(cfg_name)
        self.num_expert = cfg.num_expert = num_expert
        self.use_infer_expert = cfg.use_infer_expert = use_infer_expert
        self.per_layer_gating = cfg.per_layer_gating = per_layer_gating
        self.moe_type = cfg.moe_type = moe_type
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        assert cfg.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.mean_pool = mean_pool
        self.factor_rep = factor_rep
        self.use_moe = use_moe
        if mean_pool:
            logger.info("Uses mean-pooling for getting representations.")
        if self.factor_rep:
            logger.info("Uses factored representations.")
        self.encode_proj = None
        if projection_dim != 0:
            logger.info("Uses encode projection layer.")
            if projection_dim == -1:
                logger.info("projection_dim=%d", cfg.hidden_size)
                self.encode_proj = SimHeader(
                    cfg.hidden_size, cfg.hidden_size, dropout=dropout
                )
            else:
                logger.info("projection_dim=%d", projection_dim)
                self.encode_proj = SimHeader(
                    cfg.hidden_size, projection_dim, dropout=dropout
                )

        if use_moe:
            model_class, orig_model = model_class
            orig_encoder = orig_model.from_pretrained(cfg_name, config=cfg, **kwargs)
            self.encoder = model_class(config=cfg)
            self.encoder.load_state_dict(
                init_moe_from_pretrained_mapping(
                    orig_encoder.state_dict(), self.encoder.state_dict()
                )
            )
        else:
            self.encoder = model_class.from_pretrained(cfg_name, config=cfg, **kwargs)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        input_embeds: T = None,
        representation_token_pos=0,
        expert_id=None,
        expert_offset: int = 0,
    ) -> Tuple[T, ...]:
        if input_embeds is None:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_embeds": input_embeds,
            }
        if expert_id is not None:
            inputs["expert_id"] = expert_id

        outputs = self.encoder(**inputs)
        sequence_output = outputs[0]
        if self.encoder.config.output_hidden_states:
            hidden_states = outputs[2]
        else:
            hidden_states = None

        if self.use_moe and self.use_infer_expert:
            total_entropy_loss = outputs[-1]
        else:
            total_entropy_loss = None

        if self.mean_pool:
            mask = attention_mask.unsqueeze(-1).float()
            factor = torch.sum(attention_mask, dim=1, keepdim=True).float()
            pooled_output = torch.sum(sequence_output * mask, dim=1) / (factor + 1e-6)
        elif self.factor_rep:
            start_output = sequence_output[:, 0, :]
            bsz = sequence_output.size(0)
            max_seq_len = sequence_output.size(1)
            end_indices = torch.sum(attention_mask, dim=1).clamp_max_(max_seq_len - 1)
            mid_indices = torch.div(end_indices, 2, rounding_mode="floor")
            end_output = torch.stack(
                [sequence_output[i, end_indices[i], :] for i in range(bsz)]
            )
            mid_output = torch.stack(
                [sequence_output[i, mid_indices[i], :] for i in range(bsz)]
            )
            pooled_output = (start_output + mid_output + end_output) / 3.0
            # pooled_output = (start_output + mid_output) / 2.0
        elif isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, 0, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)

        return sequence_output, pooled_output, hidden_states, total_entropy_loss
        # return HFEncoderOutput(
        #     sequence_output=sequence_output,
        #     pooled_output=pooled_output,
        #     hidden_states=hidden_states,
        #     total_entropy_loss=total_entropy_loss,
        # )

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.encoder.config.hidden_size


def get_optimizer_v2(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    use_layer_lr: bool = True,
    n_layers: int = 12,
    layer_decay: float = 0.8,
    use_t5: bool = False,
    opt_name: str = "adam",
    moe_factor: float = 1.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    def _unpack_group_params(group_params):
        params, wdecays, layer_lrs = zip(**group_params)
        return {
            "params": params,
            "weight_decay": wdecays[0],
            "layer_lrs": layer_lrs,
        }

    if opt_name == "adam" and use_layer_lr:
        logger.info("Using Adam w layerwise adaptive learning rate")
        logger.info("Adam beta1=%f, beta2=%f", adam_betas[0], adam_betas[1])
        if use_t5:
            logger.info("Using T5")
            name_to_adapt_lr = get_layer_lrs_for_t5(
                layer_decay=layer_decay,
                n_layers=n_layers,
            )
        else:
            name_to_adapt_lr = get_layer_lrs(
                layer_decay=layer_decay,
                n_layers=n_layers,
            )
        optimizer_grouped_parameters = []
        logger.info(name_to_adapt_lr)
        for name, param in model.named_parameters():
            update_for_var = False
            for key in name_to_adapt_lr:
                if key in name:
                    update_for_var = True
                    lr_adapt_weight = name_to_adapt_lr[key]
            if not update_for_var:
                raise ValueError("No adaptive LR for %s" % name)

            wdecay = weight_decay
            if any(nd in name for nd in no_decay):
                # Parameters with no decay.
                wdecay = 0.0
            if "moe" in name:
                logger.info(f"Applying moe_factor {moe_factor} for LR with {name}")
                lr_adapt_weight *= moe_factor
            optimizer_grouped_parameters.append(
                {
                    "params": param,
                    "weight_decay": wdecay,
                    "lr_adapt_weight": lr_adapt_weight,
                }
            )
        optimizer = AdamWLayer(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_eps,
            betas=adam_betas,
        )
    elif opt_name == "adam":
        logger.info("Using Adam")
        logger.info("Adam beta1=%f, beta2=%f", adam_betas[0], adam_betas[1])
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=adam_eps,
            betas=adam_betas,
        )
    elif opt_name == "adafactor":
        logger.info("Using Adafactor")
        optimizer = Adafactor(
            model.parameters(),
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )

    return optimizer


class BertTensorizerV2(Tensorizer):
    def __init__(
        self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        if tokenizer.pad_token_id is None:
            raise ValueError("The given tokenizer has no pad_token_id!")
        self.pad_token_id = tokenizer.pad_token_id
        if tokenizer.sep_token_id is not None:
            self.eos_token_id = tokenizer.sep_token_id
        elif tokenizer.eos_token_id is not None:
            self.eos_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("The tokenizer has no special token for eos")

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                padding="max_length",
                # pad_to_max_length=True,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                padding="max_length",
                # pad_to_max_length=True,
                truncation=True,
            )
        # seq_len = self.max_length
        # if self.pad_to_max and len(token_ids) < seq_len:
        #     token_ids = token_ids + [self.pad_token_id] * (
        #         seq_len - len(token_ids)
        #     )
        # elif len(token_ids) >= seq_len:
        #     token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
        #     token_ids[-1] = self.eos_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return (tokens_tensor != self.get_pad_id()).int()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]
