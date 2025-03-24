
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

PROJECT_ID = "task2tool"
output_file = "./assets/output.png"
prompt = """Generate a wrench with a hammer head on the other end of the handle, and a screw driver perpendicular to the handle""" + """ White background, dark-colored object, 3D style"""
negative_prompt = "Text, close-up, cropped, out of frame, bad proportions"

vertexai.init(project="task2tool", location="us-central1")

model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

num_images = 3

images = model.generate_images(
    prompt=prompt,
    negative_prompt=negative_prompt,
    number_of_images=num_images,
    language="en",
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
    guidance_scale=15,
    seed=42,
    add_watermark=False
)

images[0].save(location=output_file, include_generation_parameters=False)

# Optional. View the generated image in a notebook.
# images[0].show()

print(f"Created output image using {len(images[0]._image_bytes)} bytes")
# Example response:
# Created output image using 1234567 bytes