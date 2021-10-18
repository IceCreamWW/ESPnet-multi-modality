from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import math
import random
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy, make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

import pdb

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRSwapEmbeddingModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        text_preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        text_encoder: AbsEncoder,
        shared_encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        text_in_mask_ratio: float = 0.2,
        swap_embedding_ratio: float = 0.2,
        swap_embedding_phoneme_aware: bool = True,
        swap_embedding_aug_only: bool = False,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        subsample_factor: int = 4,
        swap_embedding_position: str="before_shared",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.text_preencoder = text_preencoder
        self.encoder = encoder
        self.shared_encoder = shared_encoder
        self.text_encoder = text_encoder
        self.text_in_mask_ratio = text_in_mask_ratio
        self.swap_embedding_ratio = swap_embedding_ratio
        self.swap_embedding_position = swap_embedding_position
        self.swap_embedding_phoneme_aware = swap_embedding_phoneme_aware
        self.swap_embedding_aug_only = swap_embedding_aug_only
        self.subsample_factor = subsample_factor
        if self.swap_embedding_position == "before_encode":
            self.subsample_factor = 1
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        self.MASK, self.PAD, self.SIL = 1, 2, 3
        if ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
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

    def apply_mask_to_text_in(self, text_in: torch.Tensor, text_in_lengths: torch.Tensor):
        len_mask = make_non_pad_mask(text_in_lengths)
        with torch.no_grad():
            mask = torch.randn_like(text_in.float()).uniform_() < self.text_in_mask_ratio
            len_mask = len_mask.to(mask.device)
            mask = mask & len_mask
            text_in[mask] = self.MASK

    def expand_alignment_to_text_in(
        self,
        text_in: torch.Tensor, text_in_lengths: torch.Tensor,
        alignment: torch.Tensor, alignment_lengths: torch.Tensor,
        speech: torch.Tensor, speech_lengths: torch.Tensor
    ):
        with torch.no_grad():
            bsz = text_in.shape[0]
            text_in_expand = torch.zeros(speech.shape[:2], device=speech.device).long() + self.SIL
            for i in range(bsz):
                repeat = alignment[i][:alignment_lengths[i]]
                text_in_expand[i][:repeat.sum()] = text_in[i][:text_in_lengths[i]].repeat_interleave(repeat)
                text_in_expand[i][speech_lengths[i]:] = -1

        return text_in_expand, speech_lengths

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
                    start, end = alignment_cumsums[i][index-1] // self.subsample_factor, alignment_cumsums[i][index] // self.subsample_factor
                    mask[start: end] = 0
            else:
                mask = (torch.randn(text_embedding[i].shape[0], device=text_embedding.device).uniform() > self.swap_embedding_ratio).float()

            mask.unsqueeze_(1)
            text_embedding[i] = speech_embedding[i] * mask + text_embedding[i] * (1 - mask)

            """not working implement, overfitting"""
            # text_embedding_tmp = text_embedding[i] * mask + speech_embedding[i] * (1 - mask)
            # speech_embedding[i] = speech_embedding[i] * mask + text_embedding[i] * (1 - mask)
            # text_embedding[i] = text_embedding_tmp


    def forward(
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

        # 1.1 Pre Encode
        # 1.1a Pre Encode Speech
        speech_pre_encoder_out, speech_pre_encoder_out_lens = self.pre_encode_speech(speech, speech_lengths)
        # 1.1b Pre Encode Text
        if text_in is not None:
            if self.text_in_mask_ratio > 0:
                self.apply_mask_to_text_in(text_in, text_in_lengths)
            text_in, text_in_lengths = self.expand_alignment_to_text_in(
                    text_in, text_in_lengths, alignment, alignment_lengths, speech_pre_encoder_out, speech_pre_encoder_out_lens)

            text_in[text_in==-1] = self.PAD
            text_pre_encoder_out, text_pre_encoder_out_lens = self.pre_encode_text(text_in, text_in_lengths)

            if self.swap_embedding_position == "before_encode" and self.swap_embedding_ratio > 0:
                self.swap_embedding(speech_pre_encoder_out, text_pre_encoder_out, alignment, alignment_lengths)

        # 1.2 Encode 
        # 1.2a Encode Speech
        speech_encoder_out, speech_encoder_out_lens = self.encode_speech(speech_pre_encoder_out, speech_pre_encoder_out_lens)
        # 1.2b  Encode Text
        if text_in is not None:
            text_encoder_out, text_encoder_out_lens = self.encode_text(text_pre_encoder_out, text_pre_encoder_out_lens)
            if self.swap_embedding_position == "before_shared" and self.swap_embedding_ratio > 0:
                self.swap_embedding(speech_encoder_out, text_encoder_out, alignment, alignment_lengths)

        # 1.3 Shared Encode
        # 1.3a Shared Encode Speech
        speech_encoder_out, speech_encoder_out_lens, _ = self.shared_encoder(speech_encoder_out, speech_encoder_out_lens)
        # 1.3b Shared Encode Text
        if text_in is not None:
            text_encoder_out, text_encoder_out_lens, _ = self.shared_encoder(text_encoder_out, text_encoder_out_lens)
            if self.swap_embedding_position == "after_shared" and self.swap_embedding_ratio > 0:
                self.swap_embedding(speech_encoder_out, text_encoder_out, alignment, alignment_lengths)

        if self.swap_embedding_aug_only:
            speech_encoder_out = text_encoder_out
            text_in = None

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                speech_encoder_out, speech_encoder_out_lens, text, text_lengths
            )
            if text_in is not None:
                text_loss_att, _, _, _ = self._calc_att_loss(
                    text_encoder_out, text_encoder_out_lens, text, text_lengths
                )
                # loss_att += text_loss_att
                loss_att = (loss_att + text_loss_att) / 2

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                speech_encoder_out, speech_encoder_out_lens, text, text_lengths
            )
            if text_in is not None:
                text_loss_ctc, _ = self._calc_ctc_loss(
                    text_encoder_out, text_encoder_out_lens, text, text_lengths
                )
                # loss_ctc += text_loss_ctc
                loss_ctc = (loss_ctc + text_loss_ctc) / 2


        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
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

    def pre_encode_text(
        self, text_in: torch.Tensor, text_in_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Text PreEncoder."""
        if self.text_preencoder is not None:
            feats, feats_lengths = self.text_preencoder(text_in, text_in_lengths)
        else:
            feats, feats_lengths = text_in, text_in_lengths
        return feats, feats_lengths

    def encode_text(
        self, feats: torch.Tensor, feats_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Text Encoder.

        Args:
            text_in: (Batch, Length, ...)
            text_in_lengths: (Batch, )
        """
        encoder_out, encoder_out_lens, _ = self.text_encoder(feats, feats_lengths)

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

    def pre_encode_speech(
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

        return feats, feats_lengths

    def encode_speech(
        self, feats: torch.Tensor, feats_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def encode(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        speech_pre_encoder_out, speech_pre_encoder_out_lens = self.pre_encode_speech(speech, speech_lengths)
        speech_encoder_out, speech_encoder_out_lens = self.encode_speech(speech_pre_encoder_out, speech_pre_encoder_out_lens)
        speech_encoder_out, speech_encoder_out_lens, _ = self.shared_encoder(speech_encoder_out, speech_encoder_out_lens)
        return speech_encoder_out, speech_encoder_out_lens


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
