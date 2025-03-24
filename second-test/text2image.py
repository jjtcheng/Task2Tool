
import json
from dotenv import load_dotenv
import os
import requests
from openai import OpenAI
from pydantic import BaseModel

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

import base64


class ImageReasoning(BaseModel):
    explanation: str
    score: float

class Evaluation(BaseModel):
    final_answer: list[ImageReasoning]

class Text2Image:
    def __init__(self):
        load_dotenv()
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    # def analyze_images(self, image_urls, query, model="gpt-4o-mini"):
    #     images_content = [{"type": "image_url", "image_url": {"url": image_url, "detail": "low"}} for image_url in image_urls]
    #
    #     messages = [
    #         {"role": "system", "content": "Analyze this image for suitability in a 3D conversion pipeline and relevance of the tool to the query. Evaluate clarity, background simplicity, other distractions, object complexity, and object quality. Give score between 0 and 1. Let's think step-by-step."},
    #         {"role": "user", "content": [{"type": "text", "text": f"Query: {query}"}] + images_content
    #         }
    #     ]
    #
    #     try:
    #         completion = self.openai_client.beta.chat.completions.parse(
    #           model=model,
    #           store=True,
    #           messages=messages,
    #           response_format=Evaluation
    #         )
    #         output = completion.choices[0].message.parsed
    #         if self.DEBUG: print(output)
    #         return output
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return None


    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def analyze_images(self, image_paths, query, model="gpt-4o-mini"):
        image_data_list = [self.encode_image(image_path) for image_path in image_paths]

        messages = [
            {"role": "system", "content": "Analyze this image for how relevant / accurate the tool is to the query and suitability in a 3D conversion pipeline. Evaluate accuracy to query, clarity, background simplicity, other distractions, object complexity, and object quality. Give score between 0 and 1. Let's think step-by-step."},
            {"role": "user", "content": [{"type": "text", "text": f"Query: {query}"},
                            *[
                                 {
                                     "type": "image_url",
                                     "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                 }
                                for image_data in image_data_list
                            ]],
            }
        ]

        try:
            completion = self.openai_client.beta.chat.completions.parse(
              model=model,
              store=True,
              messages=messages,
              response_format=Evaluation
            )
            output = completion.choices[0].message.parsed
            if self.DEBUG: print(output)
            return output
        except Exception as e:
            print(f"Error: {e}")
            return None






class PixabayText2Image(Text2Image):
    def __init__(self):
        super().__init__()
        self.BASE_URL = "https://pixabay.com/api/"

    def search_images(self, query, image_type="photo", num_images=5):
        """
        Search for images on Pixabay.

        :param query: The search term ie. the tool type.
        :param image_type: Type of image to search for (default is "photo").
                           Options: "all", "photo", "illustration", "vector".
        :param per_page: Number of results per page
        :param page: number of results to fetch (default is 1).
        :return: JSON response containing search results or an error message.
        """
        params = {
            "key": os.getenv("PIXABAY_API_KEY"),
            "q": query,
            "image_type": image_type,
            "per_page": num_images,
            "page": 1
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while making API request: {e}")
            return None

    def run(self, query, num_images=5):
        # Search for relevant images
        search_results = self.search_images(query, num_images=num_images)
        if search_results is None:
            return None
        # Get all urls (use lowest resolution images)
        hits = search_results['hits']
        image_urls = [hit['previewURL'] for hit in hits]

        if self.DEBUG:
            print(f"Image URLS: {image_urls}")
        # Analyze images and rank
        evaluation = self.analyze_images(image_urls, query)
        assert len(evaluation.final_answer) == len(hits), "Evaluation number does not match number of images retrieved"

        # Find best image
        best_img_info = None
        best_score = 0.0
        for img_eval, img_info in zip(evaluation.final_answer, hits):
            if img_eval.score > best_score:
                best_score = img_eval.score
                best_img_info = img_info

        assert best_img_info is not None, "No good images"
        return best_img_info





class GoogleText2Image(Text2Image):
    def __init__(self):
        super().__init__()
        self.BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def search_images(self, query, num_images=5):
        params = {
            "key": os.getenv("GOOGLESEARCH_API_KEY"),
            "cx": os.getenv("GOOGLESEARCH_CSE_ID"),
            "q": query,
            "searchType": "image",
            "num": num_images,
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while making API request: {e}")
            return None

    def run(self, query, num_images=5):
        # Search for relevant images
        search_results = self.search_images(query, num_images=num_images)
        if search_results is None:
            return None
        # Get all urls (use lowest resolution images)
        hits = search_results['hits']
        image_urls = [hit['previewURL'] for hit in hits]

        if self.DEBUG:
            print(f"Image URLS: {image_urls}")
        # Analyze images and rank
        evaluation = self.analyze_images(image_urls, query)
        assert len(evaluation.final_answer) == len(hits), "Evaluation number does not match number of images retrieved"

        # Find best image
        best_img_info = None
        best_score = 0.0
        for img_eval, img_info in zip(evaluation.final_answer, hits):
            if img_eval.score > best_score:
                best_score = img_eval.score
                best_img_info = img_info

        assert best_img_info is not None, "No good images"
        return best_img_info




class ImagenText2Image(Text2Image):
    def __init__(self):
        super().__init__()
        vertexai.init(project="task2tool", location="us-central1")
        self.model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
        self.positive_prompt = """ White background, dark-colored object, 3D style"""
        self.negative_prompt = """Text, close-up, cropped, out of frame, bad proportions"""


    def run(self, query, num_images=3, output_dir="./assets/tmp/"):


        # Guidance scale
        #                 * 0-9 (low strength)
        #                 * 10-20 (medium strength)
        #                 * 21+ (high strength)
        images = self.model.generate_images(
            prompt=query  + self.positive_prompt,
            # negative_prompt=self.negative_prompt, # not available for imagen 3.2
            number_of_images=num_images,
            language="en",
            aspect_ratio="1:1",
            safety_filter_level="block_some",
            person_generation="dont_allow",
            guidance_scale=100,
            seed=42,
            add_watermark=False
        )

        # Save all generated images
        img_paths = []
        for i, image in enumerate(images):
            output_path = os.path.join(output_dir, f"{i+1}.jpg")
            image.save(location=output_path, include_generation_parameters=False)
            print(f"Created {output_path} using {len(image._image_bytes)} bytes")
            img_paths.append(output_path)

        # Analyze images and rank
        evaluation = self.analyze_images(img_paths, query)
        assert len(evaluation.final_answer) == len(img_paths), "Evaluation number does not match number of images retrieved"

        # Find best image
        best_img_path = None
        best_score = 0.0
        for img_eval, img_path in zip(evaluation.final_answer, img_paths):
            if img_eval.score > best_score:
                best_score = img_eval.score
                best_img_path = img_path

        return best_img_path


if __name__ == "__main__":
    text2image = ImagenText2Image()
    # query = """Generate a wrench with a hammer head on the other end of the handle, and a screw driver perpendicular to the handle."""
    query = """Extended Reach Hook"""
    print(text2image.run(query))
