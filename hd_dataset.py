# cb_train_dataset.py

from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv
import re

random.seed(0)


def find_substring(main_string, sub_string):
    """Find the start and end indices of sub_string in main_string."""
    start = main_string.find(sub_string)
    if start != -1:
        return start, start + len(sub_string)
    return -1, -1


class CircuitBreakerDataset(Dataset):

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 num_examples,
                 lorra_args,
                 model_name_or_path,
                use_only_redacted=False):  # Add this parameter
        super(CircuitBreakerDataset, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = 1024
        self.use_only_redacted = use_only_redacted  # Store this parameter
        self.tokenizer = tokenizer

        # Define tokenization kwargs
        self.tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        self.tokenize_kwargs_no_tensors = dict(max_length=1024, padding="max_length", truncation=True)

        # 修改这部分以使用 num_examples
        self.num_examples = num_examples

        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

        # ================ Model and Template Config  ================
        # Default configs
        sep_token = ""
        switch_select = [0]
        use_refusal_retain = False
        user_tag, assistant_tag = None, None
        if 'llama-3' in self.model_name_or_path:
            print("USING LLAMA 3 TEMPLATE")
            user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
            assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            switch_select = [0, 1]
            use_refusal_retain = True
        elif 'mistral' in self.model_name_or_path or 'llama-2' in self.model_name_or_path:
            print("USING MISTRAL/LLaMA 2 TEMPLATE")
            # fix spacing issue in template
            tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '  ' + message['content'] + '  ' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            user_tag = "  "
            assistant_tag = "  "
            sep_token = " "
        else:
            raise NotImplementedError(f"Config {self.model_name_or_path} not found")

        assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token

        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2:
                continue

            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=messages[1]["content"])

            orig_s.append(formatted_input)

            if len(orig_s) > num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        print("orig_s_retain[0]", orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))

        # ======================= Borderline Retain ======================= #
        # from https://github.com/paul-rottger/exaggerated-safety
        with open(f'data/xstest_v2_completions_gpt4_gpteval.csv', newline='') as f:
            data = [dict(row) for row in csv.DictReader(f)]
            data = [row for row in data if row['final_label'] == "1_full_compliance"]

        borderline_orig_s = []
        for i, d in enumerate(data * 50):
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=d['completion'])
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=d['completion'])

            borderline_orig_s.append(formatted_input)

        self.orig_s_retain += borderline_orig_s
        random.shuffle(self.orig_s_retain)
        print("borderline_orig_s[0]", borderline_orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))

        # ======================= Refusal Retain ======================= #
        if use_refusal_retain:
            with open("data/circuit_breakers_train.json") as file:
                dataset = json.load(file)

            random.shuffle(dataset)
            dataset = dataset[:2000]
            refusal_retain_orig = []
            for i, d in tqdm(enumerate(dataset * 2)):
                switch = np.random.choice(switch_select)
                if switch == 0:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag, assistant_tag=assistant_tag,
                        instruction=d['prompt'], response=d['llama3_output'])
                elif switch == 1:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag, assistant_tag=assistant_tag,
                        instruction="", response=d['llama3_output'])

                refusal_retain_orig.append(formatted_input)

            self.orig_s_retain += refusal_retain_orig
            random.shuffle(self.orig_s_retain)
            print("refusal_orig_s[0]", refusal_retain_orig[0])
            print("Orig s length:", len(self.orig_s_retain))

        # ======================= Circuit Breaker ======================= #
        with open("data/circuit_breakers_train.json") as file:
            dataset = json.load(file)
        circuit_breaker_orig = []

        for i, d in tqdm(enumerate(dataset)):
            cb_output = d['output']
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=cb_output)
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=cb_output)

            circuit_breaker_orig.append(formatted_input)

        self.circuit_breaker_orig = circuit_breaker_orig
        random.shuffle(self.circuit_breaker_orig)
        print("circuit_breaker_orig[0]", circuit_breaker_orig[0])
        print("Short circuit length:", len(self.circuit_breaker_orig))

        # ======================= Redacted Circuit Breaker with Labels ======================= #
        with open("data/redacted_circuit_breakers_train.json") as file:
            redacted_dataset = json.load(file)
        redacted_circuit_breaker_orig = []

        for d in tqdm(redacted_dataset):
            cb_output = d['output']
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=cb_output)
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=cb_output)

            redacted_circuit_breaker_orig.append({
                'text': formatted_input,
                'redacted_content': d['redacted_content'],
                'sub_category': d['category']
            })

        self.redacted_circuit_breaker_orig = redacted_circuit_breaker_orig
        print("Redacted circuit_breaker_orig[0]", redacted_circuit_breaker_orig[0])
        print("Redacted circuit breaker length:", len(self.redacted_circuit_breaker_orig))

        if self.use_only_redacted:
            self.redacted_circuit_breaker_orig = self.redacted_circuit_breaker_orig[:self.num_examples]
        else:
            self.orig_s_retain = self.orig_s_retain[:self.num_examples]
            self.circuit_breaker_orig = self.circuit_breaker_orig[:self.num_examples]
            self.redacted_circuit_breaker_orig = self.redacted_circuit_breaker_orig[:self.num_examples]

        # 打印最终的数据集大小
        print(f"Final dataset size: {len(self)}")

        # ======================= Val ======================= #
        with open("data/circuit_breakers_val.json") as file:
            dataset = json.load(file)
        val_orig = []
        for i, d in tqdm(enumerate(dataset)):
            val_orig.append(one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=d['prompt'], response=d['output']))

        self.val_orig = val_orig
        self.tokenizer = tokenizer
        # Assign eos_token as pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        

    def __len__(self):
        if self.use_only_redacted:
            return len(self.redacted_circuit_breaker_orig)
        else:
            return min(len(self.orig_s_retain), len(self.circuit_breaker_orig), len(self.redacted_circuit_breaker_orig))



    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.use_only_redacted:
            redacted_sample = self.redacted_circuit_breaker_orig[i]
            text = redacted_sample['text']
            redacted_contents = redacted_sample['redacted_content']

            # Tokenize without return_tensors, but with return_offsets_mapping
            tokenized_router = self.tokenizer(text, return_offsets_mapping=True, **self.tokenize_kwargs_no_tensors)
            input_ids = tokenized_router['input_ids']
            attention_mask = tokenized_router['attention_mask']
            offsets = tokenized_router['offset_mapping']

            # Initialize token_labels
            token_labels = [0] * len(input_ids)

            # Convert redacted_content to redacted tokens using find_substring
            for content in redacted_contents:
                start_char, end_char = find_substring(text, content)
                if start_char == -1:
                    continue
                for idx, (token_start, token_end) in enumerate(offsets):
                    if token_end <= start_char:
                        continue
                    if token_start >= end_char:
                        break
                    token_labels[idx] = 1  # Mark as harmful

            # Remove offset_mapping
            tokenized_router.pop('offset_mapping')

            # Convert input_ids and attention_mask to tensors
            tokenized_router['input_ids'] = torch.tensor(tokenized_router['input_ids'], dtype=torch.long)
            tokenized_router['attention_mask'] = torch.tensor(tokenized_router['attention_mask'], dtype=torch.long)
            token_labels = torch.tensor(token_labels, dtype=torch.float)

            return {
                'router_input_ids': tokenized_router['input_ids'],
                'router_attention_mask': tokenized_router['attention_mask'],
                'router_token_labels': token_labels,
                'sub_category': redacted_sample['sub_category']
            }
        else:

            # Get data for activator training
            orig_s_retain = self.orig_s_retain[i]
            circuit_breaker_orig = self.circuit_breaker_orig[i]
            val_orig = self.val_orig[i % len(self.val_orig)]

            cb_tokenized_kwargs = dict(max_length=512, padding='max_length', truncation=True, return_tensors="pt")
            tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
            tokenize_kwargs_no_tensors = dict(max_length=1024, padding="max_length",
                                            truncation=True)  # 不包含 return_tensors="pt"

            # =========== Circuit Breaker Inputs ===========
            cb_request, cb_response = circuit_breaker_orig.split('<SEPARATOR>')
            self.tokenizer.padding_side = "left"
            tokenized_request_circuit_breaker = self.tokenizer(cb_request, **cb_tokenized_kwargs)
            self.tokenizer.padding_side = "right"
            response_tokenized_circuit_breaker = self.tokenizer(cb_response, add_special_tokens=False,
                                                                **cb_tokenized_kwargs)
            self.tokenizer.padding_side = "left"

            combined_input_ids_circuit_breaker = torch.cat(
                [tokenized_request_circuit_breaker["input_ids"], response_tokenized_circuit_breaker["input_ids"]], dim=1)
            combined_attention_mask_circuit_breaker = torch.cat(
                [tokenized_request_circuit_breaker["attention_mask"], response_tokenized_circuit_breaker["attention_mask"]],
                dim=1)

            # ========== Retain Inputs ===========
            tokenized_inputs_retain = self.tokenizer(orig_s_retain.replace('<SEPARATOR>', self.sep_token),
                                                    **tokenize_kwargs)

            # =========== Val Inputs ===========
            tokenized_inputs_val = self.tokenizer(val_orig.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)

            # =========== Router Training Inputs ===========
            redacted_sample = self.redacted_circuit_breaker_orig[i % len(self.redacted_circuit_breaker_orig)]
            text = redacted_sample['text']
            redacted_contents = redacted_sample['redacted_content']  # 使用 redacted_content 进行匹配

            # Tokenize without return_tensors, but with return_offsets_mapping
            tokenized_router = self.tokenizer(text, return_offsets_mapping=True, **tokenize_kwargs_no_tensors)
            input_ids = tokenized_router['input_ids']
            attention_mask = tokenized_router['attention_mask']
            offsets = tokenized_router['offset_mapping']

            # Initialize token_labels
            token_labels = [0] * len(input_ids)

            # Convert redacted_content to redacted tokens using find_substring
            for content in redacted_contents:
                start_char, end_char = find_substring(text, content)
                if start_char == -1:
                    # 内容未找到，跳过
                    continue
                # 标记位于 [start_char, end_char) 范围内的所有词元
                for idx, (token_start, token_end) in enumerate(offsets):
                    if token_end <= start_char:
                        continue
                    if token_start >= end_char:
                        break
                    token_labels[idx] = 1  # 标记为有害

            # 移除 offset_mapping
            tokenized_router.pop('offset_mapping')

            # 将 input_ids 和 attention_mask 转换为张量
            tokenized_router['input_ids'] = torch.tensor(tokenized_router['input_ids'], dtype=torch.long)
            tokenized_router['attention_mask'] = torch.tensor(tokenized_router['attention_mask'],
                                                            dtype=torch.long)
            token_labels = torch.tensor(token_labels, dtype=torch.float)

            return {
                # Activator 训练数据
                'input_ids_circuit_breaker': combined_input_ids_circuit_breaker.squeeze(0),
                'attention_mask_circuit_breaker': combined_attention_mask_circuit_breaker.squeeze(0),
                'input_ids': tokenized_inputs_retain["input_ids"].squeeze(0),
                'attention_mask': tokenized_inputs_retain["attention_mask"].squeeze(0),
                'input_ids_val': tokenized_inputs_val["input_ids"].squeeze(0),
                'attention_mask_val': tokenized_inputs_val["attention_mask"].squeeze(0),
                # Router 训练数据
                'router_input_ids': tokenized_router['input_ids'],
                'router_attention_mask': tokenized_router['attention_mask'],
                'router_token_labels': token_labels,
                'sub_category': redacted_sample['sub_category']  # 添加 sub_category 字段
            }


if __name__ == "__main__":
    # 测试程序
    from transformers import AutoTokenizer

    # 假设使用 LLaMA 3 模型
    model_name_or_path = "/home/meilingrui/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873/"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)  # 替换为实际模型
    # model_name_or_path = "bert-base-uncased"  # 替换为实际模型路径
    num_examples = 5  # 加载少量样本用于测试

    # 实例化数据集
    dataset = CircuitBreakerDataset(tokenizer, num_examples, None, model_name_or_path)

    # 显示几条语料的例子
    for i in range(min(3, len(dataset))):  # 仅查看最多三条数据
        sample = dataset[i]
        original_text = tokenizer.decode(sample['input_ids_circuit_breaker'], skip_special_tokens=False)
        router_input_ids = sample['router_input_ids']
        router_token_labels = sample['router_token_labels']

        # 将 router_input_ids 转换为 token 列表
        tokens = tokenizer.convert_ids_to_tokens(router_input_ids.tolist())

        # 构建红acted 后的文本，并输出每个词元的 Router 分数
        redacted_text = ""
        for token, label in zip(tokens, router_token_labels.tolist()):
            if label == 1:
                redacted_text += "[REDACTED] "
            else:
                redacted_text += token + " "

        redacted_text = redacted_text.strip()

        print(f"Sample {i + 1}:")
        print("Original Text:")
        print(original_text)
        print("\nRedacted Text:")
        print(redacted_text)
        print("=" * 80)

        print("\nDebug: Checking proportion of 0s in token-level labels")
        num_samples_to_check = 20
        total_zeros = 0
        total_tokens = 0

        for i in range(min(num_samples_to_check, len(dataset))):
            sample = dataset[i]
            router_token_labels = sample['router_token_labels']
            
            num_zeros = (router_token_labels == 0).sum().item()
            num_tokens = len(router_token_labels)
            
            total_zeros += num_zeros
            total_tokens += num_tokens
            
            proportion_zeros = num_zeros / num_tokens
            print(f"Sample {i + 1}: Proportion of 0s = {proportion_zeros:.2f}")

        overall_proportion_zeros = total_zeros / total_tokens
        print(f"\nOverall proportion of 0s in {num_samples_to_check} samples: {overall_proportion_zeros:.2f}")