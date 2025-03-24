
from task2text import Task2Text
from text2image import ImagenText2Image


if __name__ == '__main__':
    # First stage
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
        Required features, dimensions, and materials. Provide clear, specific name suitable for direct input into search engines and use standardized tool nomenclature.
        Consideration of the 3D printer's limitations (e.g., print volume, material strength).
        If the task remains infeasible even with a new tool:
        Provide a clear explanation of the limitations preventing task completion.

        Let's think step-by-step.
    """

    task = """Move the chair closer."""
    constraints = """
    The arm of the robot is too short to reach the chair. 
    The 3D printer can only print hard objects with PLA.
    The 3D printer cannot print any tool with moving parts. 
    The 3D printer can maximally print tools of size 50cm x 50cm x 50cm.
    The 3D printer can only print the entire tool as a whole.
    Proposed tool should resemble existing tools as closely as possible.
    """
    user_prompt = """TASK: {}. CONSTRAINTS: {}""".format(task, constraints)

    output_schema_path = "./assets/output_schema.json"
    image_paths = ["./assets/room_scene.jpg"]
    model = "gpt-4o"

    task2text_agent = Task2Text()

    result = task2text_agent.run(system_prompt, user_prompt, image_paths, output_schema_path, model)

    if task2text_agent.DEBUG:
        print(result)

    feasibility_outcome = result['feasibility_check']['feasibility_outcome']
    if feasibility_outcome['outcome'] == 'tool_not_needed':
        print('No tool needed')
        exit(0)
    elif feasibility_outcome['outcome'] == 'task_infeasible':
        print('Task is simply infeasible')
        exit(0)
    else:
        print('New tool needed')
        proposed_tool = feasibility_outcome['proposed_tool']
        print(proposed_tool)

    tool_name = proposed_tool["tool_name"]
    print(f"Query is {tool_name}")


    # Second stage
    # text2image_agent = ImagenText2Image()
    # best_img_path = text2image_agent.run(query=tool_name, num_images=3, output_dir="./assets/tmp")
    # print(best_img_path)


