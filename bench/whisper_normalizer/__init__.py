# Vendored from openai/whisper v20250625 (whisper/normalizers), MIT License, Copyright (c) 2022 OpenAI.
# The text normalizer used for Whisper's published WER numbers: both the reference and the
# hypothesis go through it, so "20" vs "twenty" or "colour" vs "color" isn't counted as an error.

from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizer
