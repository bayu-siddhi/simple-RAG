import torch
import pandas as pd
from simple_RAG.helper import Helper
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available


class LLM:

    def __init__(self, model_name_or_path: str, use_quantization_config: bool, device: str) -> None:
        self.model_name_or_path: str = model_name_or_path
        self.use_quantization_config: bool = use_quantization_config
        self.device: str = Helper.set_device(device, self.__class__.__name__)

        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = 'flash_attention_2'
        else:
            attn_implementation = 'sdpa'

        print(f"[INFO] Using model: {self.model_name_or_path}")
        print(f"[INFO] Using quantization config: {self.use_quantization_config}")
        print(f"[INFO] Using attention implementation: {attn_implementation}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Raise OSError or ValueError
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path
        )

        """
        - Raise OSError or ValueError
        - If use_quantization_config = True then definitely use 'cuda'
        - torch.float16 cannot run with cpu device, source: https://github.com/huggingface/diffusers/issues/1659
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,  # datatype to use in LLM
            quantization_config=quantization_config if self.use_quantization_config else None,
            low_cpu_mem_usage=False,  # use full memory
            attn_implementation=attn_implementation  # which attention version to use
        )

        if not self.use_quantization_config:
            self.model.to(self.device)

    def generate_response(
            self,
            role: str,
            query: str,
            use_context: bool = False,
            df_context: pd.DataFrame | None = None,
            temperature: float = 0.7,
            max_new_tokens: int = 512
    ) -> str:

        if use_context:
            prompt = self.__prompt_formatter(role, query, use_context, df_context)
        else:
            prompt = self.__prompt_formatter(role, query, use_context)

        input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        if temperature != 0.0:
            outputs = self.model.generate(
                **input_ids,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens
            )
        else:
            outputs = self.model.generate(
                **input_ids,
                temperature=temperature,
                do_sample=False,
                max_new_tokens=max_new_tokens
            )

        outputs_decoded = self.tokenizer.decode(outputs[0])
        outputs_decoded = outputs_decoded.replace(prompt, "").\
            replace("<bos>", "").\
            replace("<eos>", "")

        return outputs_decoded

    def __prompt_formatter(
            self,
            role: str,
            query: str,
            use_context: bool = False,
            df_context: pd.DataFrame | None = None
    ) -> str:

        if use_context:
            # Join context items into one paragraph
            context = "- " + "\n- ".join(list(df_context['chunk']))
            # Update base prompt with context items and query
            base_prompt = Helper.base_prompt(query=query, context=context)
        else:
            base_prompt = query

        chat_template = Helper.chat_template(role, base_prompt)
        prompt = self.tokenizer.apply_chat_template(
            conversation=chat_template,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt
