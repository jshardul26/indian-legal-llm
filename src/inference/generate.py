import contextlib
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


class LegalQAGenerator:
    """
    Wraps a (base or LoRA-adapted) causal LM for text generation.

    IMPORTANT: When `model` is a PeftModel, `use_lora` controls whether the
    adapter is left enabled (fine-tuned behaviour) or temporarily disabled
    (true base-model behaviour) for a given `generate()` call. This lets two
    generators share ONE loaded model instead of loading the base weights
    twice just to compare "before" and "after" LoRA outputs.
    """

    def __init__(self, model, tokenizer, use_lora: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_lora = use_lora
        self.device = next(model.parameters()).device
        logger.info(
            "LegalQAGenerator (use_lora=%s) running on device: %s%s",
            use_lora,
            self.device,
            " -- WARNING: CPU inference will be slow (minutes/question)"
            if self.device.type == "cpu" else "",
        )

    def format_prompt(self, question: str):
        """
        Build the prompt using the tokenizer's own chat template rather
        than a hand-written "User: ... Assistant:" string.

        Qwen2.5-Instruct (like most instruct-tuned chat models) was trained
        on a specific structured format (ChatML: <|im_start|>user ...
        <|im_end|> <|im_start|>assistant). Feeding it a different, made-up
        format means it never reliably recognizes "this turn is over" and
        will ramble into fabricated follow-up Q&A, unrelated topics, or
        repeated filler until it hits max_new_tokens.

        return_dict=True is passed explicitly: depending on the installed
        transformers version, apply_chat_template with tokenize=True can
        return either a raw tensor OR a BatchEncoding/dict. Being explicit
        here avoids that ambiguity biting us (as it just did).
        """
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def _eos_token_ids(self):
        """
        The model's *actual* stop token(s) for instruct-tuned chat models
        are often not the same as tokenizer.eos_token_id alone. Qwen2.5
        defines eos_token_id as a LIST in generation_config.json (typically
        both <|im_end|> and <|endoftext|>). Passing only a single id to
        generate() means the model can emit its real stop token and never
        actually halt.
        """
        eos_ids = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            if im_end_id not in eos_ids:
                eos_ids = list(eos_ids) + [im_end_id]

        return eos_ids

    def _adapter_context(self):
        """
        If this generator is meant to represent the BASE model but is
        sharing weights with a PeftModel (i.e. an adapter is attached),
        disable the adapter just for this generation call.
        """
        is_peft_model = isinstance(self.model, PeftModel)
        if is_peft_model and not self.use_lora:
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def generate(
        self,
        question: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        encoded = self.format_prompt(question)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        attention_mask = (
            attention_mask.to(self.device)
            if attention_mask is not None
            else torch.ones_like(input_ids)
        )
        prompt_len = input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=self._eos_token_ids(),
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad(), self._adapter_context():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_tokens = outputs[0][prompt_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Belt-and-suspenders: if any stray "</s>"-style text still leaks
        # through, cut it off. Should rarely trigger now that real EOS ids
        # are passed to generate(), but costs nothing to keep.
        for stop_str in ("</s>", "<|im_end|>", "<|endoftext|>"):
            if stop_str in answer:
                answer = answer.split(stop_str, 1)[0].strip()

        return answer


def load_model_and_tokenizer(
    model_id: str,
    device_map="auto",
    load_in_4bit: bool = True,
):
    """Load base model and tokenizer."""

    common_kwargs = dict(device_map=device_map, trust_remote_code=True)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Passing attn_implementation=None explicitly can trip up some
        # transformers versions; "eager" is a safe, always-valid fallback.
        attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            **common_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            **common_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_lora_adapter(model, adapter_path: str) -> PeftModel:
    """
    Attach a LoRA adapter to `model` and return the resulting PeftModel.

    NOTE: PeftModel.from_pretrained wraps `model` IN PLACE - it swaps the
    target Linear layers inside the very same nn.Module object you pass in
    for LoraLayer replacements. If some other object (e.g. a separate
    LegalQAGenerator) keeps a direct reference to that original `model`
    object, it will silently start producing LoRA-adjusted output too,
    since the adapter is enabled by default on the shared weights. Use the
    returned PeftModel's `disable_adapter()` context (already wired up in
    LegalQAGenerator above) to get genuine base-model output instead of
    loading two full copies of the model into memory.
    """
    return PeftModel.from_pretrained(model, adapter_path)