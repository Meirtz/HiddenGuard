# hidden_guard.py

import logging
import os
import json
import gc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    HfArgumentParser,
)
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hd__dataset import CircuitBreakerDataset
from utils import save_model_and_tokenizer, get_model_generation
from args import (
    ModelArguments,
    TrainingArguments,
    LoraArguments,
    LorraArguments,
)

class Activator(nn.Module):
    def __init__(self, model, transform_layers, lora_r, lora_alpha):
        super(Activator, self).__init__()
        self.model = model
        self.transform_layers = transform_layers
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        hidden_size = self.model.config.hidden_size

        # Initialize LoRA modules for specified layers
        self.lora_modules = nn.ModuleDict()
        for layer_idx in self.transform_layers:
            self.lora_modules[str(layer_idx)] = nn.ModuleDict({
                'lora_A': nn.Linear(hidden_size, lora_r, bias=False),
                'lora_B': nn.Linear(lora_r, hidden_size, bias=False)
            })
            nn.init.normal_(self.lora_modules[str(layer_idx)]['lora_A'].weight, std=0.02)
            nn.init.zeros_(self.lora_modules[str(layer_idx)]['lora_B'].weight)

        # Attention mechanism to compute attention weights over tokens
        self.attention = nn.Linear(hidden_size, 1)

        # Activation MLP: Computes global activation from weighted hidden states
        self.activation_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states_list):
        # hidden_states_list: List of hidden states from specified layers
        delta_hidden_states = []
        for idx, hidden_states in zip(self.transform_layers, hidden_states_list):
            lora_A = self.lora_modules[str(idx)]['lora_A']
            lora_B = self.lora_modules[str(idx)]['lora_B']
            delta_hidden = lora_B(lora_A(hidden_states))
            delta_hidden_states.append(delta_hidden)

        # Combine delta_hidden_states by summing them
        combined_delta_hidden = sum(delta_hidden_states)  # Shape: [batch_size, seq_len, hidden_size]

        # Compute attention weights over tokens
        attention_weights = F.softmax(self.attention(combined_delta_hidden).squeeze(-1), dim=1)  # Shape: [batch_size, seq_len]

        # Compute weighted average of hidden states
        global_activation = torch.sum(attention_weights.unsqueeze(-1) * combined_delta_hidden, dim=1)  # Shape: [batch_size, hidden_size]

        # Compute activation signal
        activation_signal = self.activation_mlp(global_activation).squeeze(-1)  # Shape: [batch_size]
        return activation_signal  # Shape: [batch_size]


class RouterNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers=1, num_heads=2, dim_feedforward=512):
        super(RouterNetwork, self).__init__()

        # 存储参数为实例属性
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        # attention_mask: [batch_size, seq_length]
        src_key_padding_mask = ~attention_mask.bool()  # Invert mask for TransformerEncoder
        # Transformer Encoder forward pass
        encoded_states = self.transformer_encoder(
            hidden_states.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask
        )
        encoded_states = encoded_states.transpose(0, 1)  # Shape: [batch_size, seq_length, hidden_size]
        # Classification layer
        harmfulness_scores = self.classifier(encoded_states).squeeze(-1)  # Shape: [batch_size, seq_length]
        return harmfulness_scores



class HiddenStateCapture:
    def __init__(self, layers):
        self.layers = layers  # Should be a list of integers
        self.hidden_states = []
        self.hooks = []

    def attach_hooks(self, model):
        # For LLaMA 2 or Mistral models, the transformer layers are in model.model.layers
        transformer_layers = model.model.layers

        for idx, layer in enumerate(transformer_layers):
            if idx in self.layers:
                hook = layer.register_forward_hook(self.save_hidden_state)
                self.hooks.append(hook)

    def save_hidden_state(self, module, input, output):
        # The output might be a tuple depending on the model, handle accordingly
        if isinstance(output, tuple):
            self.hidden_states.append(output[0].detach())
        else:
            self.hidden_states.append(output.detach())

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset(self):
        self.hidden_states = []

def data_collator(batch_list):
    # Implement data collation function, handling Activator and Router data
    batch = {}

    # Initialize all keys
    keys = batch_list[0].keys()
    for key in keys:
        batch[key] = []

    # Collect data for each sample
    for sample in batch_list:
        for key in keys:
            batch[key].append(sample[key])

    # Convert lists to tensors and apply appropriate padding
    for key in batch:
        if key in ['input_ids_circuit_breaker', 'input_ids', 'input_ids_val', 'router_input_ids']:
            batch[key] = torch.nn.utils.rnn.pad_sequence(batch[key], batch_first=True, padding_value=0)
        elif key in ['attention_mask_circuit_breaker', 'attention_mask', 'attention_mask_val', 'router_attention_mask']:
            batch[key] = torch.nn.utils.rnn.pad_sequence(batch[key], batch_first=True, padding_value=0)
        elif key == 'router_token_labels':
            batch[key] = torch.nn.utils.rnn.pad_sequence(batch[key], batch_first=True, padding_value=0).float()
        else:
            raise ValueError(f"Unknown key {key} in data_collator")

    return batch


def evaluate(model, activator, router_network, batch, device, hidden_state_capture, transform_layers, tokenizer, context_window=3):
    """
    Evaluate the performance of Activator and Router networks on the current batch.

    Parameters:
        model: Pretrained language model.
        activator: Activator model.
        router_network: Router network.
        batch: Current training batch.
        device: Training device.
        hidden_state_capture: Instance for capturing hidden states.
        transform_layers: Layers of interest for Activator.
        tokenizer: Tokenizer.
        context_window: Number of tokens to display before and after the activated token.

    Returns:
        activator_metrics: Metrics for Activator (e.g., accuracy).
        router_metrics: Metrics for Router (e.g., F1-score).
        redacted_samples: List of samples with redacted outputs and activation details.
    """
    model.eval()
    activator.eval()
    router_network.eval()

    total_activator_correct = 0
    total_activator = 0
    all_router_preds = []
    all_router_labels = []
    redacted_samples = []

    with torch.no_grad():
        # Get Activator predictions
        input_ids_retain = batch['input_ids'].to(device)
        attention_mask_retain = batch['attention_mask'].to(device)
        input_ids_cb = batch['input_ids_circuit_breaker'].to(device)
        attention_mask_cb = batch['attention_mask_circuit_breaker'].to(device)

        # Forward pass for retain data
        hidden_state_capture.reset()
        _ = model(input_ids=input_ids_retain, attention_mask=attention_mask_retain)
        hidden_states_retain_list = hidden_state_capture.hidden_states.copy()

        # Forward pass for circuit breaker data
        hidden_state_capture.reset()
        _ = model(input_ids=input_ids_cb, attention_mask=attention_mask_cb)
        hidden_states_cb_list = hidden_state_capture.hidden_states.copy()

        # Activator signals
        activation_signal_retain = activator(hidden_states_retain_list)  # Shape: [batch_size]
        activation_signal_cb = activator(hidden_states_cb_list)          # Shape: [batch_size]

        # Labels
        activation_labels_retain = torch.zeros_like(activation_signal_retain).to(device)
        activation_labels_cb = torch.ones_like(activation_signal_cb).to(device)

        # Predictions
        activator_preds_retain = (activation_signal_retain > 0.5)
        activator_preds_cb = (activation_signal_cb > 0.5)

        # Metrics
        activator_preds = torch.cat([activator_preds_retain, activator_preds_cb], dim=0)
        activator_labels = torch.cat([activation_labels_retain > 0.5, activation_labels_cb > 0.5], dim=0)

        total_activator_correct += (activator_preds == activator_labels).sum().item()
        total_activator += activator_labels.size(0)

        # Get Router predictions
        router_input_ids = batch['router_input_ids'].to(device)
        router_attention_mask = batch['router_attention_mask'].to(device)
        router_labels = batch['router_token_labels'].to(device)

        outputs_router = model(input_ids=router_input_ids, attention_mask=router_attention_mask,
                               output_hidden_states=True)
        hidden_states_router = outputs_router.hidden_states[-1]  # Shape: [batch_size, seq_length, hidden_size]

        router_scores = router_network(hidden_states_router, router_attention_mask)  # Shape: [batch_size, seq_length]
        router_preds = (torch.sigmoid(router_scores) > 0.5).long()
        router_labels = router_labels.long()

        all_router_preds.extend(router_preds.cpu().numpy().flatten().tolist())
        all_router_labels.extend(router_labels.cpu().numpy().flatten().tolist())

        # Generate redacted outputs for the first three samples
        num_samples_to_test = min(3, router_input_ids.size(0))  # Test only up to the batch size
        for idx in range(num_samples_to_test):
            # Determine if the sample requires redaction based on Activator
            requires_redaction = activator_preds[idx].item()

            # Retrieve the corresponding router input
            router_input_id = router_input_ids[idx]
            router_attention = router_attention_mask[idx]
            router_label = router_labels[idx]

            # Decode the input text
            text = tokenizer.decode(router_input_id, skip_special_tokens=True)

            # Get router predictions for this sample
            preds = router_preds[idx].cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(router_input_id.tolist())

            # Initialize variables for first activated token
            first_activation_pos = -1  # Indicates no activation
            activation_context = ""

            if requires_redaction:
                # Find the first token predicted as harmful
                harmful_positions = torch.where(router_preds[idx] == 1)[0]
                if harmful_positions.numel() > 0:
                    first_activation_pos = harmful_positions[0].item()
                    start_idx = max(0, first_activation_pos - context_window)
                    end_idx = min(len(tokens), first_activation_pos + context_window + 1)
                    activation_context = tokenizer.decode(router_input_id[start_idx:end_idx], skip_special_tokens=True)
                else:
                    activation_context = "No harmful tokens detected despite activation signal."

            else:
                activation_context = "No activation detected."

            # Generate redacted text
            redacted_text = ""
            redact = False
            for token, pred in zip(tokens, preds):
                if pred:
                    if not redact:
                        redacted_text += "[REDACTED] "
                        redact = True
                else:
                    redacted_text += token + " "
                    redact = False
            redacted_text = redacted_text.strip()

            redacted_samples.append({
                'original_text': text,
                'redacted_text': redacted_text,
                'first_activation_pos': first_activation_pos if requires_redaction else None,
                'activation_context': activation_context
            })

    # Calculate Activator Accuracy
    activator_accuracy = total_activator_correct / total_activator if total_activator > 0 else 0.0

    # Calculate Router Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_router_labels, all_router_preds, average='binary', zero_division=0
    )
    router_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    activator_metrics = {
        'accuracy': activator_accuracy
    }

    return activator_metrics, router_metrics, redacted_samples


def train():
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()

    # Parse transform_layers: convert "10,20" into [10, 20]
    transform_layers = [int(layer.strip()) for layer in lorra_args.transform_layers.split(',')]
    print(f"Using transform_layers: {transform_layers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the LLaMA 2 or Mistral model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=AutoConfig.from_pretrained(model_args.model_name_or_path),
        cache_dir=training_args.cache_dir,
    ).to(device)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Initialize Activator with transform_layers
    activator = Activator(model, transform_layers, lora_args.lora_r, lora_args.lora_alpha).to(device)

    # Initialize RouterNetwork
    hidden_size = model.config.hidden_size
    router_network = RouterNetwork(hidden_size).to(device)

    # Load and prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )

    # Add special tokens if necessary
    special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>', '[REDACTED]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare training dataset
    train_dataset = CircuitBreakerDataset(
        tokenizer=tokenizer,
        num_examples=10000,
        lorra_args=lorra_args,
        model_name_or_path=model_args.model_name_or_path
    )
    print(f"TRAIN LEN: {len(train_dataset)}")

    # Define separate optimizers for Activator and Router
    optimizer_activator = torch.optim.AdamW(
        activator.parameters(),
        lr=training_args.learning_rate
    )
    optimizer_router = torch.optim.AdamW(
        router_network.parameters(),
        lr=training_args.learning_rate
    )

    # Define loss functions
    bce_loss_fn = nn.BCELoss()

    # Define data loader
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True,
                                  collate_fn=data_collator)

    # Attach hooks to capture hidden states from the specified layers
    hidden_state_capture = HiddenStateCapture(transform_layers)
    hidden_state_capture.attach_hooks(model)

    # Training loop
    model.eval()  # Model is in evaluation mode since we're not updating its weights
    activator.train()
    router_network.train()

    for epoch in range(int(training_args.num_train_epochs)):
        total_loss_activator = 0.0
        total_loss_router = 0.0
        for step, batch in tqdm(enumerate(train_dataloader)):
            # Zero gradients for both optimizers
            optimizer_activator.zero_grad()
            optimizer_router.zero_grad()

            # Move data to device
            input_ids_retain = batch['input_ids'].to(device)
            attention_mask_retain = batch['attention_mask'].to(device)
            input_ids_cb = batch['input_ids_circuit_breaker'].to(device)
            attention_mask_cb = batch['attention_mask_circuit_breaker'].to(device)
            router_input_ids = batch['router_input_ids'].to(device)
            router_attention_mask = batch['router_attention_mask'].to(device)
            router_labels = batch['router_token_labels'].to(device)

            # === Activator Training ===
            # Reset hidden state capture and get hidden states for retain and circuit breaker data
            hidden_state_capture.reset()
            _ = model(input_ids=input_ids_retain, attention_mask=attention_mask_retain)
            hidden_states_retain_list = hidden_state_capture.hidden_states.copy()

            hidden_state_capture.reset()
            _ = model(input_ids=input_ids_cb, attention_mask=attention_mask_cb)
            hidden_states_cb_list = hidden_state_capture.hidden_states.copy()

            # Activator forward pass
            activation_signal_retain = activator(hidden_states_retain_list)  # Shape: [batch_size]
            activation_signal_cb = activator(hidden_states_cb_list)          # Shape: [batch_size]

            # Activation loss
            activation_labels_retain = torch.zeros_like(activation_signal_retain).to(device)
            activation_labels_cb = torch.ones_like(activation_signal_cb).to(device)
            activation_loss_retain = bce_loss_fn(activation_signal_retain, activation_labels_retain)
            activation_loss_cb = bce_loss_fn(activation_signal_cb, activation_labels_cb)
            activation_loss = activation_loss_retain + activation_loss_cb

            # Backward pass and optimize Activator
            activation_loss.backward()
            optimizer_activator.step()

            total_loss_activator += activation_loss.item()

            # === Router Training ===
            # Router forward pass
            outputs_router = model(input_ids=router_input_ids, attention_mask=router_attention_mask,
                                   output_hidden_states=True)
            hidden_states_router = outputs_router.hidden_states[-1]  # Shape: [batch_size, seq_length, hidden_size]

            router_scores = router_network(hidden_states_router, router_attention_mask)  # Shape: [batch_size, seq_length]
            router_loss = bce_loss_fn(torch.sigmoid(router_scores), router_labels)

            # Backward pass and optimize Router
            router_loss.backward()
            optimizer_router.step()

            total_loss_router += router_loss.item()

            # Logging
            if step % training_args.logging_steps == 0:
                print(
                    f"Epoch [{epoch + 1}/{int(training_args.num_train_epochs)}], Step [{step}], "
                    f"Activator Loss: {activation_loss.item():.4f}, Router Loss: {router_loss.item():.4f}"
                )

            if step % 100 == 0:
                activator_metrics, router_metrics, redacted_samples = evaluate(
                    model, activator, router_network, batch, device, hidden_state_capture,
                    transform_layers, tokenizer
                )

                print(f"Step {step} Evaluation Results:")
                print(f"Activator Accuracy: {activator_metrics['accuracy'] * 100:.2f}%")
                print(f"Router Precision: {router_metrics['precision'] * 100:.2f}%, "
                      f"Recall: {router_metrics['recall'] * 100:.2f}%, "
                      f"F1 Score: {router_metrics['f1_score'] * 100:.2f}%\n")

                # Display redacted samples
                print("=== Sample Redacted Outputs ===")
                for idx, sample in enumerate(redacted_samples[:3]):  # Display first 3 samples
                    print(f"Sample {idx + 1}:")
                    print("Original Text:")
                    print(sample['original_text'])
                    print("\nRedacted Text:")
                    print(sample['redacted_text'])
                    if sample['first_activation_pos'] is not None:
                        print(f"\nFirst Activated Token Position: {sample['first_activation_pos']}")
                        print(f"Activation Context: {sample['activation_context']}")
                    else:
                        print("\nActivation Context: No activation detected.")
                    print("=" * 80)


        # Average loss for the epoch
        avg_loss_activator = total_loss_activator / len(train_dataloader)
        avg_loss_router = total_loss_router / len(train_dataloader)
        print(f"Epoch [{epoch + 1}] Average Activator Loss: {avg_loss_activator:.4f}")
        print(f"Epoch [{epoch + 1}] Average Router Loss: {avg_loss_router:.4f}")

        # Save the model after each epoch
        # save_model_and_tokenizer(
        #     model_name_or_path=model_args.model_name_or_path,
        #     model=model,
        #     tokenizer=train_dataset.tokenizer,
        #     drop_layers_after=None,
        #     output_dir=training_args.output_dir,
        #     trainer=None,
        #     activator=activator,
        #     router_network=router_network
        # )

        save_model_and_tokenizer(
            model_name_or_path=model_args.model_name_or_path,
            model=model,
            tokenizer=tokenizer,
            output_dir=training_args.output_dir,
            activator=activator,
            router_network=router_network,
            lorra_args=lorra_args  # Pass this if you have lorra_args
        )

    # After training, remove the hooks
    hidden_state_capture.remove_hooks()

if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    train()