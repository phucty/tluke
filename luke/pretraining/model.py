from typing import Dict, Optional

import torch
from luke.model import LukeConfig, LukeModel
from luke.model_table import LukeTableConfig, LukeTableModel
from luke.pretraining.metrics import Accuracy, Average
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from transformers.models.luke.modeling_luke import EntityPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class CLSEntityPredictionHead(nn.Module):
    def __init__(self, config: LukeConfig):
        super().__init__()
        self.config = config
        self.decoder = nn.Linear(config.hidden_size, config.entity_emb_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.entity_emb_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class LukePretrainingModel(LukeModel):
    def __init__(self, config: LukeConfig):
        super().__init__(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.lm_head = RobertaLMHead(config)
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.weight
        else:
            self.cls = BertPreTrainingHeads(config)
            self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        if self.config.cls_entity_prediction:
            self.cls_entity_predictions = CLSEntityPredictionHead(config)
            self.loss_fn_cls = nn.MSELoss()

        self.apply(self.init_weights)

        self.metrics = {
            "masked_lm_loss": Average(),
            "masked_lm_accuracy": Accuracy(),
            "masked_entity_loss": Average(),
            "masked_entity_accuracy": Accuracy(),
            "entity_prediction_loss": Average(),
        }

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_segment_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
        masked_entity_labels: Optional[torch.LongTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
        page_id: torch.LongTensor = None,
        **kwargs,
    ):
        model_dtype = next(self.parameters()).dtype  # for fp16 compatibility

        output = super().forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        if entity_ids is not None:
            word_sequence_output, entity_sequence_output = output[:2]
        else:
            word_sequence_output = output[0]

        ret = dict(loss=word_ids.new_tensor(0.0, dtype=model_dtype))

        if masked_entity_labels is not None:
            entity_mask = masked_entity_labels != -1
            if entity_mask.sum() > 0:
                target_entity_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
                target_entity_sequence_output = target_entity_sequence_output.view(-1, self.config.hidden_size)
                target_entity_labels = torch.masked_select(masked_entity_labels, entity_mask)

                entity_scores = self.entity_predictions(target_entity_sequence_output)
                entity_scores = entity_scores.view(-1, self.config.entity_vocab_size)

                masked_entity_loss = self.loss_fn(entity_scores, target_entity_labels)
                self.metrics["masked_entity_loss"](masked_entity_loss)
                self.metrics["masked_entity_accuracy"](
                    prediction=torch.argmax(entity_scores, dim=1), target=target_entity_labels
                )
                ret["loss"] += masked_entity_loss

        if masked_lm_labels is not None:
            masked_lm_mask = masked_lm_labels != -1
            if masked_lm_mask.sum() > 0:
                masked_word_sequence_output = torch.masked_select(word_sequence_output, masked_lm_mask.unsqueeze(-1))
                masked_word_sequence_output = masked_word_sequence_output.view(-1, self.config.hidden_size)

                if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
                    masked_lm_scores = self.lm_head(masked_word_sequence_output)
                else:
                    masked_lm_scores = self.cls.predictions(masked_word_sequence_output)
                masked_lm_scores = masked_lm_scores.view(-1, self.config.vocab_size)
                masked_lm_labels = torch.masked_select(masked_lm_labels, masked_lm_mask)

                masked_lm_loss = self.loss_fn(masked_lm_scores, masked_lm_labels)

                self.metrics["masked_lm_loss"](masked_lm_loss)
                self.metrics["masked_lm_accuracy"](
                    prediction=torch.argmax(masked_lm_scores, dim=1), target=masked_lm_labels
                )
                ret["loss"] += masked_lm_loss

        if page_id is not None:
            true_entity_embeddings = torch.gather(
                self.entity_embeddings.entity_embeddings.weight,
                0,
                page_id.unsqueeze(1).repeat(1, self.config.entity_emb_size),
            )
            cls_token_embeddings = word_sequence_output[:, 0]
            pred_entity_embeddings = self.cls_entity_predictions(cls_token_embeddings)
            entity_prediction_loss = self.loss_fn_cls(pred_entity_embeddings, true_entity_embeddings)
            ret["loss"] += entity_prediction_loss
            self.metrics["entity_prediction_loss"](entity_prediction_loss)

        return ret

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {k: m.get_metric(reset=reset) for k, m in self.metrics.items()}


class CellEmbeddingHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.entity_emb_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class EntityHorRelPredictionHead(LukeTableModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transform = CellEmbeddingHeadTransform(config)
        # 4 cells, --> 2: Is same relationship or not
        self.decoder = nn.Linear(config.entity_emb_size * 4, 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(2))
        # self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states.view(hidden_states.size(0), -1)) + self.bias
        return hidden_states


class EntityVerRelPredictionHead(LukeTableModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transform = CellEmbeddingHeadTransform(config)
        # 4 cells, --> 2: Is same relationship or not
        self.decoder = nn.Linear(config.entity_emb_size * 4, 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(2))
        # self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states.view(hidden_states.size(0), -1)) + self.bias
        return hidden_states


class WordHorRelPredictionHead(LukeTableModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transform = CellEmbeddingHeadTransform(config)
        # 4 cells, --> 2: Is same relationship or not
        self.decoder = nn.Linear(config.entity_emb_size * 4, 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(2))
        # self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states.view(hidden_states.size(0), -1)) + self.bias
        return hidden_states


class WordVerRelPredictionHead(LukeTableModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transform = CellEmbeddingHeadTransform(config)
        # 4 cells, --> 2: Is same relationship or not
        self.decoder = nn.Linear(config.entity_emb_size * 4, 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(2))
        # self.apply(self.init_weights)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states.view(hidden_states.size(0), -1)) + self.bias
        return hidden_states


class LukeTablePretrainingModel(LukeTableModel):
    def __init__(self, config: LukeTableConfig):
        super().__init__(config)

        if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
            self.lm_head = RobertaLMHead(config)
            self.lm_head.decoder.weight = self.embeddings.word_embeddings.weight
        else:
            self.cls = BertPreTrainingHeads(config)
            self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight

        self.entity_predictions = EntityPredictionHead(config)
        self.entity_predictions.decoder.weight = self.entity_embeddings.entity_embeddings.weight

        self.ent_hor_rel_predictions = EntityHorRelPredictionHead(config)
        self.ent_ver_rel_predictions = EntityVerRelPredictionHead(config)

        self.word_hor_rel_predictions = WordHorRelPredictionHead(config)
        self.word_ver_rel_predictions = WordVerRelPredictionHead(config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        if self.config.cls_entity_prediction:
            self.cls_entity_predictions = CLSEntityPredictionHead(config)
            self.loss_fn_cls = nn.MSELoss()

        self.apply(self.init_weights)

        self.metrics = {
            "masked_lm_loss": Average(),
            "masked_lm_accuracy": Accuracy(),
            "masked_entity_loss": Average(),
            "masked_entity_accuracy": Accuracy(),
            "entity_prediction_loss": Average(),
            "relationship_accuracy": Accuracy(),
            "relationship_loss": Average(),
            "ent_ver_rel_loss": Average(),
            "ent_ver_rel_acc": Accuracy(),
            "ent_hor_rel_loss": Average(),
            "ent_hor_rel_acc": Accuracy(),
            "word_ver_rel_loss": Average(),
            "word_ver_rel_acc": Accuracy(),
            "word_hor_rel_loss": Average(),
            "word_hor_rel_acc": Accuracy(),
        }

    def forward(
        self,
        word_ids: torch.LongTensor,
        word_row_ids: torch.LongTensor,
        word_col_ids: torch.LongTensor,
        word_segment_ids: torch.LongTensor,
        word_attention_mask: torch.LongTensor,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        entity_position_row_ids: torch.LongTensor = None,
        entity_position_col_ids: torch.LongTensor = None,
        entity_segment_ids: torch.LongTensor = None,
        entity_attention_mask: torch.LongTensor = None,
        masked_entity_labels: Optional[torch.LongTensor] = None,
        masked_lm_labels: Optional[torch.LongTensor] = None,
        page_id: torch.LongTensor = None,
        ent_hor_rel_positions: torch.LongTensor = None,
        ent_hor_rel_labels: torch.LongTensor = None,
        ent_ver_rel_positions: torch.LongTensor = None,
        ent_ver_rel_labels: torch.LongTensor = None,
        word_hor_rel_positions: torch.LongTensor = None,
        word_hor_rel_labels: torch.LongTensor = None,
        word_ver_rel_positions: torch.LongTensor = None,
        word_ver_rel_labels: torch.LongTensor = None,
        **kwargs,
    ):
        model_dtype = next(self.parameters()).dtype  # for fp16 compatibility

        output = super().forward(
            word_ids=word_ids,
            word_row_ids=word_row_ids,
            word_col_ids=word_col_ids,
            word_segment_ids=word_segment_ids,
            word_attention_mask=word_attention_mask,
            entity_ids=entity_ids,
            entity_position_ids=entity_position_ids,
            entity_position_row_ids=entity_position_row_ids,
            entity_position_col_ids=entity_position_col_ids,
            entity_segment_ids=entity_segment_ids,
            entity_attention_mask=entity_attention_mask,
        )

        if entity_ids is not None:
            word_sequence_output, entity_sequence_output = output[:2]
        else:
            word_sequence_output = output[0]

        ret = dict(loss=word_ids.new_tensor(0.0, dtype=model_dtype))

        if masked_entity_labels is not None:
            entity_mask = masked_entity_labels != -1
            if entity_mask.sum() > 0:
                target_sequence_output = torch.masked_select(entity_sequence_output, entity_mask.unsqueeze(-1))
                target_sequence_output = target_sequence_output.view(-1, self.config.hidden_size)
                target_entity_labels = torch.masked_select(masked_entity_labels, entity_mask)

                entity_scores = self.entity_predictions(target_sequence_output)
                entity_scores = entity_scores.view(-1, self.config.entity_vocab_size)

                masked_entity_loss = self.loss_fn(entity_scores, target_entity_labels)
                self.metrics["masked_entity_loss"](masked_entity_loss)
                self.metrics["masked_entity_accuracy"](
                    prediction=torch.argmax(entity_scores, dim=1), target=target_entity_labels
                )
                ret["loss"] += masked_entity_loss

        if masked_lm_labels is not None:
            masked_lm_mask = masked_lm_labels != -1
            if masked_lm_mask.sum() > 0:
                masked_word_sequence_output = torch.masked_select(word_sequence_output, masked_lm_mask.unsqueeze(-1))
                masked_word_sequence_output = masked_word_sequence_output.view(-1, self.config.hidden_size)

                if self.config.bert_model_name and "roberta" in self.config.bert_model_name:
                    masked_lm_scores = self.lm_head(masked_word_sequence_output)
                else:
                    masked_lm_scores = self.cls.predictions(masked_word_sequence_output)
                masked_lm_scores = masked_lm_scores.view(-1, self.config.vocab_size)
                masked_lm_labels = torch.masked_select(masked_lm_labels, masked_lm_mask)

                masked_lm_loss = self.loss_fn(masked_lm_scores, masked_lm_labels)

                self.metrics["masked_lm_loss"](masked_lm_loss)
                self.metrics["masked_lm_accuracy"](
                    prediction=torch.argmax(masked_lm_scores, dim=1), target=masked_lm_labels
                )
                ret["loss"] += masked_lm_loss

        if ent_hor_rel_labels is not None:
            mask_labels = ent_hor_rel_labels != -1
            if mask_labels.sum() > 0:
                target_labels = torch.masked_select(ent_hor_rel_labels, mask_labels)
                # dk: number of valid samples (relationship)
                dk = target_labels.size(0)

                # db: batch size
                # dn: number of token
                # dh: hidden size
                db, dn, dh = entity_sequence_output.shape
                # de: number of samples
                # dc: number of cells (fixed number = 4 cells)
                # dm: max number of tokens in one cell (fixed number = 30)
                db, de, dc, dm = ent_hor_rel_positions.shape

                I_mask = (ent_hor_rel_positions != -1).type_as(entity_sequence_output)
                I_mask_sum = I_mask.sum(dim=-1).unsqueeze(-1)
                if (I_mask_sum != 0).sum() / dc == dk:
                    I2 = ent_hor_rel_positions.clamp(min=0).view(db, -1, 1).expand(db, de * dc * dm, dh)
                    cell_hiddens = (torch.gather(entity_sequence_output, 1, I2) * I_mask.view(db, -1, 1)).view(
                        db, de, dc, dm, dh
                    ).sum(dim=-2) / I_mask_sum.clamp(min=1)
                    cell_hiddens = torch.masked_select(cell_hiddens, I_mask_sum.expand(db, de, dc, dh) != 0)
                    cell_hiddens = cell_hiddens.view(-1, dc, dh)
                    scores = self.ent_hor_rel_predictions(cell_hiddens)
                    ent_hor_rel_loss = self.loss_fn(scores, target_labels)
                    self.metrics["ent_hor_rel_loss"](ent_hor_rel_loss)
                    self.metrics["ent_hor_rel_acc"](prediction=torch.argmax(scores, dim=1), target=target_labels)
                    ret["loss"] += ent_hor_rel_loss

        if ent_ver_rel_labels is not None:
            mask_labels = ent_ver_rel_labels != -1
            if mask_labels.sum() > 0:
                target_labels = torch.masked_select(ent_ver_rel_labels, mask_labels)
                # dk: number of valid samples (relationship)
                dk = target_labels.size(0)
                # db: batch size
                # dn: number of token
                # dh: hidden size
                db, dn, dh = entity_sequence_output.shape
                # de: number of samples
                # dc: number of cells (fixed number = 4 cells)
                # dm: max number of tokens in one cell (fixed number = 30)
                db, de, dc, dm = ent_ver_rel_positions.shape

                I_mask = (ent_ver_rel_positions != -1).type_as(entity_sequence_output)
                I_mask_sum = I_mask.sum(dim=-1).unsqueeze(-1)

                if (I_mask_sum != 0).sum() / dc == dk:
                    I2 = ent_ver_rel_positions.clamp(min=0).view(db, -1, 1).expand(db, de * dc * dm, dh)
                    cell_hiddens = (torch.gather(entity_sequence_output, 1, I2) * I_mask.view(db, -1, 1)).view(
                        db, de, dc, dm, dh
                    ).sum(dim=-2) / I_mask_sum.clamp(min=1)
                    cell_hiddens = torch.masked_select(cell_hiddens, I_mask_sum.expand(db, de, dc, dh) != 0)
                    cell_hiddens = cell_hiddens.view(-1, dc, dh)
                    scores = self.ent_ver_rel_predictions(cell_hiddens)
                    ent_ver_rel_loss = self.loss_fn(scores, target_labels)
                    self.metrics["ent_ver_rel_loss"](ent_ver_rel_loss)
                    self.metrics["ent_ver_rel_acc"](prediction=torch.argmax(scores, dim=1), target=target_labels)
                    ret["loss"] += ent_ver_rel_loss

        if word_hor_rel_labels is not None:
            mask_labels = word_hor_rel_labels != -1
            if mask_labels.sum() > 0:
                target_labels = torch.masked_select(word_hor_rel_labels, mask_labels)
                # dk: number of valid samples (relationship)
                dk = target_labels.size(0)
                # db: batch size
                # dn: number of token
                # dh: hidden size
                db, dn, dh = word_sequence_output.shape
                # de: number of samples
                # dc: number of cells (fixed number = 4 cells)
                # dm: max number of tokens in one cell (fixed number = 30)
                db, de, dc, dm = word_hor_rel_positions.shape

                I_mask = (word_hor_rel_positions != -1).type_as(word_sequence_output)
                I_mask_sum = I_mask.sum(dim=-1).unsqueeze(-1)

                if (I_mask_sum != 0).sum() / dc == dk:
                    I2 = word_hor_rel_positions.clamp(min=0).view(db, -1, 1).expand(db, de * dc * dm, dh)
                    cell_hiddens = (torch.gather(word_sequence_output, 1, I2) * I_mask.view(db, -1, 1)).view(
                        db, de, dc, dm, dh
                    ).sum(dim=-2) / I_mask_sum.clamp(min=1)
                    cell_hiddens = torch.masked_select(cell_hiddens, I_mask_sum.expand(db, de, dc, dh) != 0)
                    cell_hiddens = cell_hiddens.view(-1, dc, dh)
                    scores = self.word_hor_rel_predictions(cell_hiddens)
                    word_hor_rel_loss = self.loss_fn(scores, target_labels)
                    self.metrics["word_hor_rel_loss"](word_hor_rel_loss)
                    self.metrics["word_hor_rel_acc"](prediction=torch.argmax(scores, dim=1), target=target_labels)
                    ret["loss"] += word_hor_rel_loss

        if word_ver_rel_labels is not None:
            mask_labels = word_ver_rel_labels != -1
            if mask_labels.sum() > 0:
                target_labels = torch.masked_select(word_ver_rel_labels, mask_labels)
                dk = target_labels.size(0)
                # Get dimention
                # db: batch size
                # dn: number of token
                # dh: hidden size
                db, dn, dh = word_sequence_output.shape
                # de: number of samples
                # dc: number of cells (fixed number = 4 cells)
                # dm: max number of tokens in one cell (fixed number = 30)
                db, de, dc, dm = word_ver_rel_positions.shape

                I_mask = (word_ver_rel_positions != -1).type_as(word_sequence_output)
                I_mask_sum = I_mask.sum(dim=-1).unsqueeze(-1)
                if (I_mask_sum != 0).sum() / dc == dk:
                    I2 = word_ver_rel_positions.clamp(min=0).view(db, -1, 1).expand(db, de * dc * dm, dh)
                    cell_hiddens = (torch.gather(word_sequence_output, 1, I2) * I_mask.view(db, -1, 1)).view(
                        db, de, dc, dm, dh
                    ).sum(dim=-2) / I_mask_sum.clamp(min=1)
                    cell_hiddens = torch.masked_select(cell_hiddens, I_mask_sum.expand(db, de, dc, dh) != 0)

                    cell_hiddens = cell_hiddens.view(-1, dc, dh)
                    scores = self.word_ver_rel_predictions(cell_hiddens)
                    word_ver_rel_loss = self.loss_fn(scores, target_labels)
                    self.metrics["word_ver_rel_loss"](word_ver_rel_loss)
                    self.metrics["word_ver_rel_acc"](prediction=torch.argmax(scores, dim=1), target=target_labels)
                    ret["loss"] += word_ver_rel_loss

        if page_id is not None:
            true_entity_embeddings = torch.gather(
                self.entity_embeddings.entity_embeddings.weight,
                0,
                page_id.unsqueeze(1).repeat(1, self.config.entity_emb_size),
            )
            cls_token_embeddings = word_sequence_output[:, 0]
            pred_entity_embeddings = self.cls_entity_predictions(cls_token_embeddings)
            entity_prediction_loss = self.loss_fn_cls(pred_entity_embeddings, true_entity_embeddings)
            ret["loss"] += entity_prediction_loss
            self.metrics["entity_prediction_loss"](entity_prediction_loss)

        return ret

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {k: m.get_metric(reset=reset) for k, m in self.metrics.items()}
