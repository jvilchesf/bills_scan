# augmentation.py
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import numpy as np


class ReceiptAugmenter:
    """Data augmentation specifically designed for receipt images"""

    def __init__(self, training_mode=True, augmentation_prob=0.8):
        """
        Initialize the augmenter

        Args:
            training_mode: If True, apply augmentations. If False, only apply standard preprocessing
            augmentation_prob: Probability of applying augmentation to each image
        """
        self.training_mode = training_mode
        self.augmentation_prob = augmentation_prob

    def __call__(self, image):
        """Apply augmentation pipeline to image"""
        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            if hasattr(image, "convert"):
                image = image
            else:
                return image

        # Always apply basic preprocessing
        image = self._ensure_rgb(image)
        image = self._resize_maintain_aspect(image, target_size=512)

        # Apply augmentations only in training mode
        if self.training_mode and random.random() < self.augmentation_prob:
            # Apply augmentations in random order
            augmentations = [
                self._random_rotation,
                self._random_brightness,
                self._random_contrast,
                self._random_sharpness,
                self._random_noise,
                self._random_perspective,
                self._random_shadow,
            ]

            # Randomly select 2-4 augmentations to apply
            num_augmentations = random.randint(2, 4)
            selected_augmentations = random.sample(augmentations, num_augmentations)

            for aug_func in selected_augmentations:
                image = aug_func(image)

        # Final enhancement for all images
        image = self._enhance_text_clarity(image)

        return image

    def _ensure_rgb(self, image):
        """Convert image to RGB if needed"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _resize_maintain_aspect(self, image, target_size=512):
        """Resize image maintaining aspect ratio"""
        # Calculate aspect ratio
        width, height = image.size
        aspect = width / height

        if aspect > 1:  # Wider than tall
            new_width = target_size
            new_height = int(target_size / aspect)
        else:  # Taller than wide
            new_height = target_size
            new_width = int(target_size * aspect)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Pad to square if needed
        if new_width != target_size or new_height != target_size:
            padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
            x = (target_size - new_width) // 2
            y = (target_size - new_height) // 2
            padded.paste(image, (x, y))
            image = padded

        return image

    def _random_rotation(self, image):
        """Apply random small rotation (receipts are often slightly tilted)"""
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)  # Small rotation for receipts
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        return image

    def _random_brightness(self, image):
        """Adjust brightness randomly (simulates different lighting conditions)"""
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.7, 1.3)  # Avoid too dark or too bright
            image = enhancer.enhance(factor)
        return image

    def _random_contrast(self, image):
        """Adjust contrast randomly (simulates faded or strong print)"""
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.7, 1.4)
            image = enhancer.enhance(factor)
        return image

    def _random_sharpness(self, image):
        """Adjust sharpness (simulates camera focus variations)"""
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            factor = random.uniform(0.5, 1.5)
            image = enhancer.enhance(factor)
        return image

    def _random_noise(self, image):
        """Add slight gaussian noise (simulates camera sensor noise)"""
        if random.random() > 0.3:  # Less frequent
            # Convert to numpy array
            img_array = np.array(image)

            # Add gaussian noise
            noise = np.random.normal(0, 3, img_array.shape)  # Small noise
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            image = Image.fromarray(img_array)
        return image

    def _random_perspective(self, image):
        """Simulate slight perspective distortion (camera angle)"""
        if random.random() > 0.4:
            width, height = image.size

            # Define small perspective shifts
            shift = random.randint(5, 15)

            # Random perspective type
            perspective_type = random.choice(["top", "bottom", "left", "right"])

            if perspective_type == "top":
                coeffs = [1, 0, 0, 0, 1, 0, -shift / width / 100, 0]
            elif perspective_type == "bottom":
                coeffs = [1, 0, 0, 0, 1, 0, shift / width / 100, 0]
            elif perspective_type == "left":
                coeffs = [1, 0, 0, 0, 1, 0, 0, -shift / height / 100]
            else:  # right
                coeffs = [1, 0, 0, 0, 1, 0, 0, shift / height / 100]

            image = image.transform(
                (width, height),
                Image.PERSPECTIVE,
                coeffs,
                Image.BICUBIC,
                fillcolor=(255, 255, 255),
            )
        return image

    def _random_shadow(self, image):
        """Add random shadow/gradient (simulates uneven lighting)"""
        if random.random() > 0.6:  # Less frequent
            width, height = image.size

            # Create gradient overlay
            gradient = Image.new("L", (width, height), 255)
            gradient_array = np.array(gradient)

            # Random gradient direction
            direction = random.choice(["horizontal", "vertical", "diagonal"])

            if direction == "horizontal":
                for x in range(width):
                    value = int(255 - (x / width) * 50)  # Max 50 darkness
                    gradient_array[:, x] = value
            elif direction == "vertical":
                for y in range(height):
                    value = int(255 - (y / height) * 50)
                    gradient_array[y, :] = value
            else:  # diagonal
                for y in range(height):
                    for x in range(width):
                        value = int(255 - ((x + y) / (width + height)) * 50)
                        gradient_array[y, x] = value

            gradient = Image.fromarray(gradient_array)

            # Apply gradient as overlay
            image = Image.blend(image, image.convert("L").convert("RGB"), 0.1)

        return image

    def _enhance_text_clarity(self, image):
        """Final enhancement to ensure text remains readable"""
        # Slight sharpening to enhance text edges
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        # Slight contrast boost for text
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)

        return image


def augment_dataset(dataset, training_mode=True):
    """
    Apply augmentation to a dataset

    Args:
        dataset: HuggingFace dataset
        training_mode: Whether to apply augmentations

    Returns:
        Augmented dataset
    """
    augmenter = ReceiptAugmenter(training_mode=training_mode)

    def augment_sample(sample):
        """Augment a single sample"""
        if "image" in sample:
            sample["image"] = augmenter(sample["image"])
        return sample

    # Apply augmentation
    augmented_dataset = dataset.map(
        augment_sample, desc=f"{'Augmenting' if training_mode else 'Processing'} images"
    )

    return augmented_dataset
