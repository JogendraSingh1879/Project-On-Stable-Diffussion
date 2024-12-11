# Project-On-Stable-Diffussion

# Using - runwayml/stable-diffusion-v1-5
To perform inference with the runwayml/stable-diffusion-v1-5 model using Python, you can utilize the diffusers library by Hugging Face. Here's a simple Python script to generate images from text prompts using this specific model.
# Explanation:
StableDiffusionPipeline: This is the core class for generating images from text prompts.
use_auth_token: A Hugging Face token is required to authenticate and access the pre-trained model.
torch.no_grad(): Used to disable gradient calculation to save memory and computation during inference.
pipe(prompt): Generates the image based on the input text prompt.
image.show(): Displays the generated image.
image.save(): Saves the generated image to a file.

This script should work seamlessly, allowing you to generate images based on text prompts using the runwayml/stable-diffusion-v1-5 model.
Stable Diffusion is a deep learning model designed for generating images from textual descriptions. It's based on a diffusion process, where noise is gradually added to an image and then removed to generate a new image from a prompt. It leverages Latent Diffusion Models (LDMs) to work in a compressed latent space, optimizing both efficiency and quality.

# Key Versions:

Stable Diffusion v1: Launched in 2022, it produced high-quality images from text prompts, becoming open-source.
Stable Diffusion v2: Released in 2022, it introduced improved models, higher resolution, and inpainting features.
Stable Diffusion v2.1: A refined version that offered better text-to-image generation, especially for detailed and coherent outputs.

Coding Sample:
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the pre-trained model and tokenizer from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)

# Move the model to GPU if available for faster inference
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Define the text prompt
prompt = "A futuristic cityscape with neon lights at night"

# Generate the image
with torch.no_grad():
    image = pipe(prompt).images[0]

# Show the generated image
image.show()

# Optionally, save the image
image.save("generated_image.png")
