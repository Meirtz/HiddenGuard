import os
import json
import torch
from transformers import LlavaNextForConditionalGeneration, AutoModelForCausalLM
import logging
import os
import json
import torch
from sklearn.metrics import precision_recall_fscore_support

def save_model_and_tokenizer(model_name_or_path, model, tokenizer, output_dir,
                             activator=None, router_network=None, lorra_args=None):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")

    # Save the main model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save activator with metadata
    if activator is not None:
        activator_path = os.path.join(output_dir, "activator.pth")
        activator_save_dict = {
            'state_dict': activator.state_dict(),
            'transform_layers': activator.transform_layers,
            'lora_r': activator.lora_r,
            'lora_alpha': activator.lora_alpha
        }
        torch.save(activator_save_dict, activator_path)
        print(f"Activator saved to {activator_path}")

    # Save router_network with metadata
    if router_network is not None:
        router_network_path = os.path.join(output_dir, "router_network.pth")
        router_network_save_dict = {
            'state_dict': router_network.state_dict(),
            'hidden_size': router_network.hidden_size,
            'num_layers': router_network.num_layers,
            'num_heads': router_network.num_heads,
            'dim_feedforward': router_network.dim_feedforward
        }
        torch.save(router_network_save_dict, router_network_path)
        print(f"Router Network saved to {router_network_path}")

    # Save lorra_args if provided
    if lorra_args is not None:
        lorra_config_path = os.path.join(output_dir, "lorra_config.json")
        with open(lorra_config_path, "w", encoding="utf-8") as file:
            json.dump(lorra_args.to_dict(), file, indent=2)
        print(f"lorra_args saved to {lorra_config_path}")
    else:
        print("No lorra_args provided; skipping saving lorra_config.json")

def save_llava_model_and_tokenizer(model_name_or_path, model, processor, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"MModel and processor saving to {output_dir}")
    
    # merge lora
    merged_model = model.merge_and_unload() 
    # merge original layers
    
    anchor_model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=merged_model.dtype)
    merged_model.language_model.model.layers = merged_model.language_model.model.layers + anchor_model.language_model.model.layers[drop_layers_after+1:]
    merged_model.config = anchor_model.config

    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()



def get_model_generation(inputs, model, tokenizer, activator, router_network, device, hidden_state_capture,
                         transform_layers, max_length=2048):
    model.eval()
    activator.eval()
    router_network.eval()

    # Prepare the conversation history
    conversation = tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False)

    # Tokenize the conversation
    input_ids = tokenizer.encode(conversation, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Store original tokenized conversation for teacher forcing
    original_input_ids = input_ids.clone()

    generated_ids = []  # Store generated ids
    redacted_output = []  # Store final redacted output

    # Track whether the activator has been triggered
    activator_triggered = False

    # Teacher forcing loop
    max_steps = original_input_ids.size(1)  # Ensure we don't exceed the length of the original input_ids
    for step in range(min(max_length, max_steps)):  # Ensure we don't exceed max_length or original_input_ids length
        # Reset hidden state capture
        hidden_state_capture.reset()

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
            next_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            hidden_states_list = hidden_state_capture.hidden_states.copy()

        # Apply activator to hidden states if activator hasn't already been triggered
        if not activator_triggered:
            activation_signal = activator(hidden_states_list)
            print(f"Activation signal: {activation_signal}")

            # Check if the activator has triggered based on the last token's activation signal
            activator_triggered = activation_signal >= 0.0  # Shape: [batch_size]

        if activator_triggered:
            # Apply the RouterNetwork to decide if the token should be redacted
            last_hidden_state = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]
            last_attention_mask = attention_mask[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]

            # RouterNetwork forward pass
            harmfulness_score = router_network(last_hidden_state, last_attention_mask)  # Shape: [batch_size, 1]
            harmfulness_score = torch.sigmoid(harmfulness_score)

            if harmfulness_score.item() >= 0.4:
                redacted_output.append("[REDACTED]")
            else:
                # Sample the next token and append it to the redacted output
                next_token_id = original_input_ids[:, step].item()  # Teacher forcing: use the ground truth token
                redacted_output.append(tokenizer.decode([next_token_id], skip_special_tokens=True))
        else:
            # Activator not triggered: generate normally, but still teacher forcing
            next_token_id = original_input_ids[:, step].item()
            redacted_output.append(tokenizer.decode([next_token_id], skip_special_tokens=True))

        # Append the generated token id for tracking
        generated_ids.append(original_input_ids[:, step].item())

        # Update input_ids and attention_mask for the next step
        if step + 1 < original_input_ids.size(1):  # Ensure we don't go out of bounds
            input_ids = original_input_ids[:, :step + 2]
            attention_mask = torch.ones_like(input_ids)

        # End generation if the EOS token is encountered
        if original_input_ids[:, step].item() == tokenizer.eos_token_id:
            break

    # Convert the generated tokens to text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    redacted_output_text = " ".join(redacted_output)

    return {
        'generated_text': generated_text,
        'redacted_output_text': redacted_output_text,
        'activation_triggered': activator_triggered.item()
    }






