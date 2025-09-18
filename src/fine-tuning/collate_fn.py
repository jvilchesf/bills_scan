from PIL import Image
import io
from .config import Config

config = Config()


def create_collate_fn(processor):
    """
    The main idea is to format the text and images for the processor
    processor will read them in a tensor format
    """

    def collate_fn(batch):
        # Get first sample
        sample = batch[0]
        # Format the sample here instead of during mapping
        #       formatted_sample = format_dataset(sample)  #
        # get the text from the sample first
        messages = sample["messages"]

        # Apply chat template to get the full conversation text
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        images = []
        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if content["type"] == "image":
                        image_data = content["image"]

                        # Convert to PIL Image based on format
                        if isinstance(image_data, dict) and "bytes" in image_data:
                            image_bytes = image_data["bytes"]
                            image = Image.open(io.BytesIO(image_bytes))
                        elif hasattr(image_data, "convert"):  # Already a PIL Image
                            image = image_data
                        else:
                            # Try to open as bytes
                            image = Image.open(io.BytesIO(image_data))

                        # Add image to a list
                        images.append(image)

        batch_data = processor(
            text=text, images=images, return_tensors="pt", padding=True
        )
        labels = create_masked_labels(
            text, batch_data["input_ids"], processor.tokenizer
        )
        # Debug: Check for issues
        if labels.sum() == -100 * labels.numel():  # All tokens masked
            print(f"WARNING: All tokens masked in sample!")
        # Debug: Check token distribution
        total_tokens = (labels != -100).sum().item()
        masked_tokens = (labels == -100).sum().item()

        if total_tokens < 10:
            print(f"WARNING: Very few training tokens ({total_tokens})")

        batch_data["labels"] = labels

        return batch_data

    return collate_fn


# Mask only system/user prompts, train on full assistant response
def create_masked_labels(full_text, input_ids, tokenizer):
    """
    Mask everything except the assistant's response for training.
    This ensures we only train on the model's outputs, not the prompts.
    """
    labels = input_ids.clone()

    # Find the assistant response in the text
    assistant_markers = [
        "assistant\n",
        "assistant:",
        "<|im_start|>assistant\n",
        "### Assistant\n",
    ]

    assistant_start_pos = -1
    assistant_marker_used = None

    for marker in assistant_markers:
        if marker in full_text:
            assistant_start_pos = full_text.find(marker) + len(marker)
            assistant_marker_used = marker
            break

    if assistant_start_pos == -1:
        print(f"WARNING: No assistant marker found in text. Not masking any tokens.")
        # Don't mask anything - train on everything (not ideal)
        labels[labels == tokenizer.pad_token_id] = -100
        return labels

    # Get the text before and including the assistant marker
    text_before_response = full_text[:assistant_start_pos]

    # Tokenize just this prefix to find how many tokens to mask
    prefix_tokens = tokenizer(
        text_before_response, add_special_tokens=False, return_tensors="pt"
    )

    num_prefix_tokens = prefix_tokens.input_ids.shape[1]

    # Mask all tokens before the assistant response
    # Handle batch dimension properly
    if labels.dim() == 1:
        labels[:num_prefix_tokens] = -100
    else:  # batch dimension
        labels[0, :num_prefix_tokens] = -100

    # Also mask padding tokens
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    # Debug output
    total_tokens = labels.numel()
    masked_tokens = (labels == -100).sum().item()
    unmasked_tokens = total_tokens - masked_tokens

    if unmasked_tokens < 10:
        print(f"WARNING: Only {unmasked_tokens} tokens to train on!")
        print(f"Text before response length: {len(text_before_response)}")
        print(f"Full text length: {len(full_text)}")
        print(
            f"Assistant response: {full_text[assistant_start_pos : assistant_start_pos + 100]}..."
        )

    return labels
