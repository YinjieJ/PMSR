# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn

from fairseq import metrics, utils, modules
from fairseq.criterions import register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, ignore_label=0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        # self.k = 0
        # self.features = None
        # self.labels = None

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # if not self.k:
        #     self.features = features
        #     self.labels = labels
        # else:
        #     self.features = torch.concat([self.features, features], dim=0)
        #     self.labels = torch.concat([self.labels, labels], dim=0)
        # self.k += 1
        # if self.k == 150:
        #     torch.save({'features':self.features.cpu().detach(), 'labels':self.labels.cpu().detach()}, '/home/yinjiejiang/workspace/pretrain/tsne.pt')
        #     return None

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-18)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(loss)

        return loss

@register_criterion("label_smoothed_cross_entropy_with_masked_lm")
class LabelSmoothedCrossEntropyCriterionWithMaskedLM(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, alignment_lambda):
        super().__init__(task, sentence_avg, label_smoothing)
        self.alignment_lambda = alignment_lambda
        self.contractive_loss_computer = SupConLoss()
        self.num_updates = 0
        self.linear = nn.Linear(768*2, 2048)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--alignment-lambda",
            default=0.05,
            type=float,
            metavar="D",
            help="weight for the alignment loss",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        assert sample['masked_source'].size(0) == sample["target"].size(0)

        masked_tokens = sample["masked_source"].ne(self.padding_idx)
        masked_sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).

        if masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        masked_targets = sample["masked_source"]
        assert masked_tokens is not None
        if masked_tokens is not None:
            masked_targets = masked_targets[masked_tokens]
        net_output = model(**sample["net_input"], masked_tokens=masked_tokens)

        # loss for decoder
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(
                0) if self.sentence_avg else sample["ntokens"]
        )

        assert masked_sample_size > 0
        # loss for encoder
        masked_loss = self.compute_masked_loss(masked_targets, net_output)

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "masked_loss": utils.item(masked_loss.data) if reduce else masked_loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "masked_sample_size": int(masked_sample_size.detach().cpu()),
        }
        # print(logging_output)
        alignment_loss = None

        # Compute alignment loss only for training set and non dummy batches.
        if "alignments" in sample and sample["alignments"] is not None:
            alignment_loss = self.compute_alignment_loss(sample, net_output)

        if alignment_loss is not None:
            logging_output["alignment_loss"] = utils.item(alignment_loss.data)
            loss += self.alignment_lambda * alignment_loss
        
        if 'label' in sample and sample['label'] is not None:
            contractive_loss = self.compute_contractive_loss(model, sample, net_output)
            logging_output["contractive_loss"] = utils.item(contractive_loss.data) if reduce else contractive_loss.data

        loss += masked_loss
        loss += 0.1 * contractive_loss * sample_size
        # loss = contractive_loss * sample_size
        # loss = masked_loss
        return loss, sample_size, logging_output

    def compute_alignment_loss(self, sample, net_output):
        attn_prob = net_output[1]["attn"][0]
        bsz, tgt_sz, src_sz = attn_prob.shape
        attn = attn_prob.view(bsz * tgt_sz, src_sz)

        align = sample["alignments"]
        align_weights = sample["align_weights"].float()

        if len(align) > 0:
            # Alignment loss computation. align (shape [:, 2]) contains the src-tgt index pairs corresponding to
            # the alignments. align_weights (shape [:]) contains the 1 / frequency of a tgt index for normalizing.
            loss = -(
                (attn[align[:, 1][:, None], align[:, 0][:, None]]).log()
                * align_weights[:, None]
            ).sum()
        else:
            return None

        return loss

    def compute_masked_loss(self, targets, net_output):

        encoder_logits = net_output[1]['masked_encoder_out'][0]
        assert encoder_logits.size(0) == targets.size(0), (encoder_logits.size(), targets.size())
        loss = modules.cross_entropy(
            encoder_logits.view(-1, encoder_logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        return loss

    def compute_contractive_loss(self, model, sample, net_output):
        labels = sample['label']
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask = (mask.sum(1) > 1) # & (labels.squeeze(-1) != 0)
        if mask.sum().int() == 0:
            return torch.zeros(1)[0]
        mask = mask.nonzero().squeeze(-1)
        enc_src = (net_output[1]['encoder_out'].transpose(0, 1))[mask]
        emb = net_output[1]['x']
        source = sample["net_input"]['src_tokens']
        src_len = sample["net_input"]['src_lengths'][mask]
        src_mask = source.eq(model.encoder.padding_idx)
        enc_src = enc_src * ((1 - src_mask.unsqueeze(-1).type_as(enc_src)))[mask]
        target = model.get_targets(sample, net_output)
        tgt_mask = target.eq(model.encoder.padding_idx) + target.eq(model.encoder.padding_idx+1)
        emb = emb * (1-tgt_mask.unsqueeze(-1).type_as(emb))
        enc_tgt = emb[mask]
        tgt_len = (~tgt_mask).sum(1)[mask].unsqueeze(1)
        # dec_output = model.encoder.contractive_forward(emb[mask], target[mask])
        # tgt_len = dec_output['src_lengths'][0]-1
        # enc_tgt = dec_output['encoder_out'][0].transpose(0, 1)
        # enc_tgt = enc_tgt * ((1-tgt_mask.unsqueeze(-1).type_as(emb))[mask])
        features = torch.cat((enc_src.sum(dim=1)/src_len.unsqueeze(1), enc_tgt.sum(dim=1)/tgt_len), dim=-1).unsqueeze(1)
        return self.contractive_loss_computer(features = self.linear(features), labels = labels.view(-1)[mask])



    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0)
                                  for log in logging_outputs))
        masked_loss_sum = utils.item(
            sum(log.get("masked_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        alignment_loss_sum = utils.item(
            sum(log.get("alignment_loss", 0) for log in logging_outputs)
        )
        contractive_loss_sum = utils.item(
            sum(log.get("contractive_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0)
                                 for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        masked_sample_size = utils.item(
            sum(log.get("masked_sample_size", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "masked_loss", masked_loss_sum / masked_sample_size / math.log(2), masked_sample_size, round=3
        )
        metrics.log_scalar(
            "contractive_loss", contractive_loss_sum  / math.log(2), 1, round=3
        )
        # metrics.log_scalar(
        #     "alignment_loss",
        #     alignment_loss_sum / sample_size / math.log(2),
        #     sample_size,
        #     round=3,
        # )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

# @register_criterion("masked_lm")
# class MaskedLmLoss(FairseqCriterion):
#     """
#     Implementation for the loss used in masked language model (MLM) training.
#     """

#     def __init__(self, task, tpu=False):
#         super().__init__(task)
#         self.tpu = tpu

#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.

#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         masked_tokens = sample["target"].ne(self.padding_idx)
#         sample_size = masked_tokens.int().sum()

#         # Rare: when all tokens are masked, project all tokens.
#         # We use torch.where to avoid device-to-host transfers,
#         # except on CPU where torch.where is not well supported
#         # (see github.com/pytorch/pytorch/issues/26247).
#         if self.tpu:
#             masked_tokens = None  # always project all tokens on TPU
#         elif masked_tokens.device == torch.device("cpu"):
#             if not masked_tokens.any():
#                 masked_tokens = None
#         else:
#             masked_tokens = torch.where(
#                 masked_tokens.any(),
#                 masked_tokens,
#                 masked_tokens.new([True]),
#             )

#         logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
#         targets = model.get_targets(sample, [logits])
#         if masked_tokens is not None:
#             targets = targets[masked_tokens]

#         logging_output = {
#             "loss": loss if self.tpu else loss.data,
#             "ntokens": sample["ntokens"],
#             "nsentences": sample["nsentences"],
#             "sample_size": sample_size,
#         }
#         return loss, sample_size, logging_output

#     @staticmethod
#     def reduce_metrics(logging_outputs) -> None:
#         """Aggregate logging outputs from data parallel training."""
#         loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
#         sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

#         metrics.log_scalar(
#             "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
#         )
#         metrics.log_derived(
#             "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
#         )

#     @staticmethod
#     def logging_outputs_can_be_summed() -> bool:
#         """
#         Whether the logging outputs returned by `forward` can be summed
#         across workers prior to calling `reduce_metrics`. Setting this
#         to True will improves distributed training speed.
#         """
#         return True
