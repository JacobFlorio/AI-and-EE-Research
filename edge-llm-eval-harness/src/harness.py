"""Hardware-aware eval harness for quantized LLMs.

The goal is to measure capability degradation *per quantization step*
and *per hardware backend* with a single command. Backends are pluggable;
every backend exposes the same `generate(prompt, max_new_tokens)`.

Current backends:
  - hf_fp16  : HuggingFace CUDA FP16 reference
  - hf_int8  : bitsandbytes 8-bit on CUDA
  - hf_int4  : bitsandbytes 4-bit (nf4) on CUDA

Planned: llama.cpp Q4_K_M / Q2_K, TensorRT-LLM INT8, Jetson INT8,
sister-project FPGA systolic array.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import time
import torch


class Backend(Protocol):
    name: str
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str: ...
    def logprob_of_continuation(self, prompt: str, continuation: str) -> float: ...


@dataclass
class HFBackend:
    model_id: str
    quant: str = "fp16"  # one of: fp16, int8, int4
    _model: object = None
    _tok: object = None

    @property
    def name(self) -> str:
        return f"hf_{self.quant}"

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs = {"device_map": "auto"}
        if self.quant == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif self.quant == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quant == "int4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            raise ValueError(f"unknown quant: {self.quant}")
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        return self

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        ids = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self._tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    def logprob_of_continuation(self, prompt: str, continuation: str) -> float:
        full = prompt + continuation
        ids = self._tok(full, return_tensors="pt").to(self._model.device)
        prompt_len = self._tok(prompt, return_tensors="pt").input_ids.shape[1]
        with torch.no_grad():
            logits = self._model(**ids).logits[0]
        # Upcast to fp32 before log_softmax so the accumulated sum of
        # token log-probs doesn't get snapped to fp16's coarse grid at
        # large magnitudes (where fp16 precision is ~0.25). Without this,
        # fp16 and int8 backends report *identical* logprobs even though
        # their logits differ, which makes the whole eval look insensitive.
        logits = logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        cont_ids = ids.input_ids[0, prompt_len:]
        token_lps = log_probs[prompt_len - 1 : -1].gather(-1, cont_ids[:, None]).squeeze(-1)
        return float(token_lps.double().sum().item())


def time_generate(backend: Backend, prompt: str, max_new_tokens: int = 64):
    t0 = time.perf_counter()
    out = backend.generate(prompt, max_new_tokens=max_new_tokens)
    dt = time.perf_counter() - t0
    return out, dt
