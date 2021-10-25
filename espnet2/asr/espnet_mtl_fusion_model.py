from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy, make_non_pad_mask, pad_list
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.mse_ctc import MSE_CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

import numpy as np
import random
import math

import pdb

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRMTLFusionModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        vocab_size_in: int,
        token_list: Union[Tuple[str, ...], List[str]],
        token_list_in: Union[Tuple[str, ...], List[str]],
        token_expand_stats: Dict[str, Tuple[float, float]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        text_encoder: AbsEncoder,
        shared_encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        mse_ctc: Optional[Union[MSE_CTC, CTC]],
        bert_ctc: Optional[Union[MSE_CTC, CTC]],
        rnnt_decoder: None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        subsample_factor: int = 4,
        swap_embedding_ratio: float = 0,
        text_in_mask_ratio: float = 0.2,
        mse_ctc_weight: float = 0.0,
        bert_ctc_weight: float = 0.0,
        swap_embedding_phoneme_aware: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.vocab_size_in = vocab_size_in
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.shared_encoder = shared_encoder
        self.text_encoder = text_encoder

        self.text_in_mask_ratio = text_in_mask_ratio
        self.swap_embedding_ratio = swap_embedding_ratio
        self.subsample_factor = subsample_factor
        self.mse_ctc_weight = mse_ctc_weight
        self.bert_ctc_weight = bert_ctc_weight
        self.token2id = {token:i for i,token in enumerate(token_list_in)}
        self.swap_embedding_phoneme_aware = swap_embedding_phoneme_aware
#         self.token_expand_stats = {self.token2id[key]:(value[0] / subsample_factor, value[1] / self.subsample_factor)
#                                             for key,value in token_expand_stats.items()}

        self.make_token_expand_stats(token_expand_stats)
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.mse_ctc = mse_ctc
        self.bert_ctc = bert_ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.MASK = 2
        self.PAD = 3
        self.SIL = 4


    def make_token_expand_stats(self, token_expand_stats):
        self.token_expand_stats = {}
        for token, id_ in self.token2id.items():
            if token in token_expand_stats:
                self.token_expand_stats[id_] = token_expand_stats[token]
            else:
                if "_" in token:
                    default_stats = token_expand_stats["DEFAULT_" + token.split('_')[1]]
                    self.token_expand_stats[id_] = default_stats


    def apply_mask_to_text_in(self, text_in: torch.Tensor, text_in_lengths: torch.Tensor):
        len_mask = make_non_pad_mask(text_in_lengths)
        with torch.no_grad():
            mask_mask = torch.randn_like(text_in.float()).uniform_() < self.text_in_mask_ratio
            len_mask = len_mask.to(mask_mask.device)
            mask_mask = mask_mask & len_mask
            text_in[mask_mask] = self.MASK

            # mask_rand = torch.randn_like(text_in.float()).uniform_() > 1 - self.text_in_mask_ratio * 0.1
            # rand = torch.randint_like(text_in, self.SIL+1, self.vocab_size_in)
            # mask_rank = mask_rand & len_mask
            # text_in = text_in * (1 - mask_rand) + rand * mask_rand

    def expand_alignment_to_text_in(
        self,
        text_in: torch.Tensor, text_in_lengths: torch.Tensor,
        alignment: torch.Tensor, alignment_lengths: torch.Tensor,
        speech: torch.Tensor, speech_lengths: torch.Tensor,
    ):

        with torch.no_grad():
            bsz = text_in.shape[0]
            text_in_expand = torch.zeros(speech.shape[:2], device=speech.device).long() + self.SIL
            for i in range(bsz):
                repeat = alignment[i][:alignment_lengths[i]]
                text_in_expand[i][:repeat.sum()] = text_in[i][:text_in_lengths[i]].repeat_interleave(repeat)
                text_in_expand[i][speech_lengths[i]:] = self.PAD

        return text_in_expand, speech_lengths

    def expand_dummy_to_text_in(self, text_in, text_in_lengths, token_expand_stats, text_in_ori):
        with torch.no_grad():
            bsz = text_in.shape[0]
            text_in_expand = []
            for i in range(bsz):
                repeat = torch.Tensor([max(round(np.random.normal(token_expand_stats[text_in_ori[i][j]][0],
                                        token_expand_stats[text_in_ori[i][j]][1])),1) for j in range(text_in_lengths[i])]).long().to(text_in.device)
                text_in_expand.append(text_in[i][:text_in_lengths[i]].repeat_interleave(repeat))
            text_in_expand_lengths = torch.Tensor([len(t) for t in text_in_expand]).long()
            text_in_expand = pad_list(text_in_expand, self.PAD)

        return text_in_expand, text_in_expand_lengths

    def swap_embedding(self, speech_embedding: torch.Tensor, text_embedding: torch.Tensor, alignment: torch.Tensor, alignment_lengths):
        assert speech_embedding.shape == text_embedding.shape

        bsz = alignment.shape[0]
        alignment_cumsums = torch.cumsum(alignment, dim=1)
        for i in range(bsz):
            if self.swap_embedding_phoneme_aware:
                mask = torch.ones(text_embedding[i].shape[0], device=text_embedding.device)
                indices = random.sample(list(range(1, alignment_lengths[i])),
                        math.ceil(alignment_lengths[i] * self.swap_embedding_ratio))
                for index in indices:
                    start, end = alignment_cumsums[i][index-1], alignment_cumsums[i][index]
                mask[start: end] = 0
            else:
                mask = (torch.randn(text_embedding[i].shape[0], device=text_embedding.device).uniform() > self.swap_embedding_ratio).float()

            mask.unsqueeze_(1)
            # text_embeddint[i] = speech_embedding[i] * mask + text_embedding[i] * (1 - mask)

            text_embedding_tmp = text_embedding[i] * mask + speech_embedding[i] * (1 - mask)
            speech_embedding[i] = speech_embedding[i] * mask + text_embedding[i] * (1 - mask)
            text_embedding[i] = text_embedding_tmp

    def forward(
        self,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        text_in_para: torch.Tensor = None,
        text_in_para_lengths: torch.Tensor = None,
        alignment: torch.Tensor = None,
        alignment_lengths: torch.Tensor = None,
        text_in: torch.Tensor = None,
        text_in_lengths: torch.Tensor = None,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        _type: str = 'speech'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if "speech" in _type:
            return self.forward_speech(speech, speech_lengths,
                                    text_in_para, text_in_para_lengths,
                                    alignment, alignment_lengths,
                                    text, text_lengths)
        else:
            return self.forward_text(text_in, text_in_lengths, text, text_lengths)

    def forward_speech(
        self,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        text_in: torch.Tensor = None,
        text_in_lengths: torch.Tensor = None,
        alignment: torch.Tensor = None,
        alignment_lengths: torch.Tensor = None,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape

        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        speech_encoder_out, speech_encoder_out_lens, speech_feats, speech_feats_lens = self.encode_speech(speech, speech_lengths)

        if text_in is not None:
            text_in_ori, text_in_lengths_ori = text_in.clone(), text_in_lengths.clone()
            if self.text_in_mask_ratio > 0:
                self.apply_mask_to_text_in(text_in, text_in_lengths)

            text_in, text_in_lengths = self.expand_alignment_to_text_in(text_in, text_in_lengths, alignment, alignment_lengths, speech_feats, speech_feats_lens)

            text_encoder_out, text_encoder_out_lens = self.encode_text(text_in, text_in_lengths)

            if self.swap_embedding_ratio > 0:
                alignment_cumsum = ((torch.cumsum(alignment, dim=1)  - 1) // (self.subsample_factor // 2) - 1) // (self.subsample_factor // 2)
                alignment[:,0] = alignment_cumsum[:,0]
                alignment[:,1:] = (alignment_cumsum[:,1:] - alignment_cumsum[:,:-1])

                self.swap_embedding(speech_encoder_out, text_encoder_out, alignment, alignment_lengths)

            text_shared_encoder_out, text_shared_encoder_out_lens, _ = self.shared_encoder(text_encoder_out, text_encoder_out_lens)

            speech_shared_encoder_out, speech_shared_encoder_out_lens = speech_encoder_out, speech_encoder_out_lens

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                speech_shared_encoder_out, speech_shared_encoder_out_lens, text, text_lengths
            )
#             if text_in is not None:
#                 text_loss_att, _, _, _ = self._calc_att_loss(
#                     text_shared_encoder_out, text_shared_encoder_out_lens, text, text_lengths
#                 )
#                 # loss_att += text_loss_att
#                 loss_att = (loss_att + text_loss_att) / 2

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                speech_shared_encoder_out, speech_shared_encoder_out_lens, text, text_lengths
            )
#             if text_in is not None:
#                 text_loss_ctc, _ = self._calc_ctc_loss(
#                     text_shared_encoder_out, text_shared_encoder_out_lens, text, text_lengths
#                 )
#                 # loss_ctc += text_loss_ctc
#                 loss_ctc = (loss_ctc + text_loss_ctc) / 2

        if text_in is None or self.mse_ctc_weight == 0.0:
            loss_mse_ctc, cer_mse_ctc = None, None
        else:
            loss_mse_ctc, cer_mse_ctc = self._calc_mse_ctc_loss(
                speech_encoder_out, speech_encoder_out_lens, text_in_ori, text_in_lengths_ori
            )


        # 2c. BERT CTC branch
        if text_in is None or self.bert_ctc_weight == 0.0:
            loss_bert_ctc, cer_bert_ctc = None, None
        else:
            loss_bert_ctc, cer_bert_ctc = self._calc_bert_ctc_loss(
                text_encoder_out, text_encoder_out_lens, text_in_ori, text_in_lengths_ori
            )


        # 2d. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if text_in is not None and self.mse_ctc_weight != 0:
            loss += self.mse_ctc_weight * loss_mse_ctc

        if text_in is not None and self.bert_ctc_weight != 0:
            loss += self.bert_ctc_weight * loss_bert_ctc

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_mse_ctc=loss_mse_ctc.detach() if loss_mse_ctc is not None else None,
            loss_bert_ctc=loss_bert_ctc.detach() if loss_bert_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def forward_text(
        self,
        text_in: torch.Tensor = None,
        text_in_lengths: torch.Tensor = None,
        text: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape

        """
        if self.text_in_mask_ratio > 0:
            self.apply_mask_to_text_in(text_in, text_in_lengths)
        text_in[text_in==-1] = self.PAD
        """

        text_in_ori_np = text_in.clone().cpu().numpy()
        text_in_ori, text_in_lengths_ori = text_in.clone(), text_in_lengths.clone()
        if self.text_in_mask_ratio > 0:
            self.apply_mask_to_text_in(text_in, text_in_lengths)
        text_in, text_in_lengths = self.expand_dummy_to_text_in(text_in, text_in_lengths, self.token_expand_stats, text_in_ori_np)
        _input = text_in
        _input_lengths = text_in_lengths

        # Check that batch_size is unified
        assert (
            _input.shape[0]
            == _input_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (_input.shape, _input_lengths.shape, text.shape, text_lengths.shape)
        batch_size = _input.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode_text(_input, _input_lengths)
        encoder_out, encoder_out_lens, _ = self.shared_encoder(encoder_out, encoder_out_lens)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        if self.bert_ctc_weight == 0.0:
            loss_bert_ctc, cer_bert_ctc = None, None
        else:
            loss_bert_ctc, cer_bert_ctc = self._calc_bert_ctc_loss(
                encoder_out, encoder_out_lens, text_in_ori, text_in_lengths_ori
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.bert_ctc_weight != 0:
            loss = loss + self.bert_ctc_weight * loss_bert_ctc

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_bert_ctc=loss_bert_ctc.detach() if loss_bert_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        text_in: torch.Tensor = None,
        text_in_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if speech is not None:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
            return {"feats": feats, "feats_lengths": feats_lengths}
        else:
            raise RuntimeError("no speech available, other input should not compute feats")
            # return {"feats": text_in, "feats_lengths": text_in_lengths}


    def encode_text(
        self, text_in: torch.Tensor, text_in_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Text Encoder.

        Args:
            text_in: (Batch, Length, ...)
            text_in_lengths: (Batch, )
        """
        encoder_out, encoder_out_lens, _ = self.text_encoder(text_in, text_in_lengths)

        assert encoder_out.size(0) == text_in.size(0), (
            encoder_out.size(),
            text_in.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def encode_speech(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens, feats, feats_lengths


    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_bert_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.bert_ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.mse_ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_mse_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.mse_ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.mse_ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError
