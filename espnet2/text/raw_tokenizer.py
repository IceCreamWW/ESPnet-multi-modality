from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import warnings

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class RawTokenizer(AbsTokenizer):
    def __init__(self):
        pass

    def text2tokens(self, line: str) -> List[str]:
        return line.strip().split()

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = " ".join(tokens)
