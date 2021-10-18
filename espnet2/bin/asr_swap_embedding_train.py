#!/usr/bin/env python3
from espnet2.tasks.asr_swap_embedding import ASRSwapEmbeddingTask


def get_parser():
    parser = ASRSwapEmbeddingTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRSwapEmbeddingTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
