from typing import Any

import tiktoken
from transformers import LlamaTokenizer


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            # 手动映射无法在 tiktoken 中自动识别的模型到编码
            custom_model_to_encoding = {
                "deepseek-chat": "cl100k_base",
            }
            normalized = model_name.strip().lower()
            # 1) 先尝试按模型名自动匹配
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                return
            except Exception:
                pass
            # 2) 再尝试自定义映射（精确或前缀匹配）
            for key, enc in custom_model_to_encoding.items():
                if normalized == key or normalized.startswith(key.split("-", 1)[0]):
                    self.tokenizer = tiktoken.get_encoding(enc)
                    return
            # 3) 最后回退到常见编码
            for enc in ["cl100k_base", "o200k_base", "gpt2"]:
                try:
                    self.tokenizer = tiktoken.get_encoding(enc)
                    return
                except Exception:
                    continue
            # 若仍失败，则抛出错误
            raise KeyError(
                f"No available tokenizer encoding for model '{model_name}'."
            )
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
