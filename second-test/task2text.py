
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
import os


class Task2Text:
    def __init__(self):
        load_dotenv()
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def get_output_schema(self, output_schema_path):
        with open(output_schema_path, 'r') as schema_file:
          return json.load(schema_file)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def create_completion(self, system_prompt, user_prompt, image_paths, json_schema, model):
        image_data_list = [self.encode_image(image_path) for image_path in image_paths]
        messages = [
                    {"role": "system", "content": system_prompt},  # Added system prompt here
                    {"role": "user",
                     "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                            *[
                                 {
                                     "type": "image_url",
                                     "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                 }
                                for image_data in image_data_list
                            ],
                        ],
                     }
                    ]


        try:
            completion = self.client.chat.completions.create(
              model=model,
              store=True,
              messages=messages,
              response_format={"type": "json_schema", "json_schema": json_schema}
            )
            output = completion.choices[0].message.content
            output_to_dict = json.loads(output)
            return output_to_dict
        except Exception as e:
            print(f"Error: {e}")
            return None

    def run(self, system_prompt, user_prompt, image_paths, output_schema_path, model):
        output_schema = self.get_output_schema(output_schema_path)
        result = self.create_completion(system_prompt, user_prompt, image_paths, output_schema, model)
        return result



if __name__ == "__main__":
    system_prompt = """
        You are a robot equipped with 3D printing capabilities.
        Objective to complete: Complete the following task: {TASK}
        Constraints that you must follow: {CONSTRAINTS}
        You are provided with images of the scene.
    
        Your mission:
            Scene Analysis:
                Provide a detailed description of the scene, including: Object types, materials, and sizes. Positions and spatial relationships between objects.
            Feasibility Check:
                Determine if the task can be completed using the current setup without printing a new tool.
                Consider reach, force, spatial constraints, and other mechanical limitations.
                Assess whether existing tools are sufficient.
            Decision Flow:
                If the task is feasible with the existing setup:
                State that no additional tool is needed.
        If the task is infeasible with the current setup:
        Clearly explain why (e.g., reach limitations, insufficient force, spatial constraints).
        If 3D printing a new tool could enable task completion then feasibility check sets new_tool_required:
        Propose a detailed design for the tool that follows the printing constraints, including:
        Required features, dimensions, and materials.
        Consideration of the 3D printer's limitations (e.g., print volume, material strength).
        If the task remains infeasible even with a new tool:
        Provide a clear explanation of the limitations preventing task completion.
    
        Let's think step-by-step.
    """


    task = """Move the chair closer."""
    constraints = """
    The arm of the robot is too short to reach the chair. 
    The 3D printer can only print hard objects with PLA and importantly it can have no moving parts. 
    The 3D printer can maximally print items of size 50cm x 50cm x 50cm.
    Tool should resemble existing tools as closely as possible.
    """
    user_prompt = """TASK: {}. CONSTRAINTS: {}""".format(task, constraints)


    output_schema_path = "./assets/output_schema.json"
    image_paths = ["./assets/room_scene.jpg"]
    model = "gpt-4o-mini"

    agent = Task2Text()

    result = agent.run(system_prompt, user_prompt, image_paths, output_schema_path, model)

    if agent.DEBUG:
        print(result)

    feasibility_outcome = result['feasibility_check']['feasibility_outcome']
    if feasibility_outcome['outcome'] == 'tool_not_needed':
        print('No tool needed')
    elif feasibility_outcome['outcome'] == 'task_infeasible':
        print('Task is simply infeasible')
    else:
        print('New tool needed')
        proposed_tool = feasibility_outcome['proposed_tool']
        print(proposed_tool)

