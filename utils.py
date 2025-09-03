import os
import json
import time
from datetime import timedelta
def load_config():
    with open("./configs/info.json", "r", encoding="utf-8") as json_file:
        info = json.load(json_file)

    with open("./configs/model_params.json", "r", encoding="utf-8") as param_file:
        model_params = json.load(param_file)

    return info, model_params

def generate_message(user_prompt, system_prompt, message_history=None):
    if message_history is not None:
        # Include conversation history if provided
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": message_history}
        ]
    else:
        # Exclude conversation history if not provided
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

def get_response_from_model(client, user_prompt, system_prompt, response_format, message_history=None, **api_params):
    # Generate the message payload
    max_retries = 1
    retry_count = 0
    while retry_count < max_retries:
        try:
            messages = generate_message(user_prompt, system_prompt, message_history)

            # Send the message to the model and get a response ChatCompletion.create
            # gpt4o계열: .beta.chat.completions.parse / chat.completions.create
            completion = client.beta.chat.completions.parse(
                messages=messages,
                response_format=response_format,
                **api_params
            )

            # Extract and return the parsed response
            result = completion.choices[0].message.parsed
            if result is None:
                return "None"
            else:
                return completion.choices[0].message.parsed

        except Exception as e:
            print(f"JSONDecodeError: {e}. Retrying {retry_count + 1}/{max_retries}...")
            retry_count += 1
            time.sleep(5) 
        
    print("Failed to parse response after maximum retries.")
    return {"error": "Failed to parse response after maximum retries"}


def process_step(step_num, client, user_prompt, system_prompt, response_format, message_history, json_config, results_dir, api_params, additional_instruction=None):
    full_prompt = f"{user_prompt} {additional_instruction}" if additional_instruction != None else user_prompt

    if message_history is not None:
        response = get_response_from_model(
            client, full_prompt, system_prompt, response_format, '\n'.join(message_history), **api_params
        )
        message_history.append(f"User: {user_prompt}\nAnswer: {str(response)}\n")
    else:
        response = get_response_from_model(
            client, full_prompt, system_prompt, response_format, **api_params
        )

    add_to_json(step_num, user_prompt, response, json_config, results_dir)
    return response

def save_granular_risk_factor(risk_factor, granular_risk_factor, granular_results_dir):
    # Ensure the results directory exists
    if not os.path.exists(granular_results_dir):
        os.makedirs(granular_results_dir)

    # Generate the JSON file name based on the risk factor
    abbr_risk_factor = risk_factor_abbreviation_map(risk_factor)  # Sanitize the file name
    json_file = os.path.join(granular_results_dir, f"{abbr_risk_factor}_grf.json")

    # Load existing data or initialize
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            data = {"risk_factor": risk_factor, "granular_risk_factors": {}}
    else:
        data = {"risk_factor": risk_factor, "granular_risk_factors": {}}

    # Add or update the granular risk factor count
    if granular_risk_factor in data["granular_risk_factors"]:
        data["granular_risk_factors"][granular_risk_factor] += 1
    else:
        data["granular_risk_factors"][granular_risk_factor] = 1

    # Save updated data back to the JSON file
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    
def generate_instruction(risk_factor, results_dir, max_attempts=5):
    # Generate the JSON file name based on the risk factor
    sanitized_risk_factor = risk_factor.replace(" ", "_").lower()
    json_file = os.path.join(results_dir, f"{sanitized_risk_factor}_grf.json")

    # Load existing granular risk factors from the JSON file
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            data = {"risk_factor": "", "granular_risk_factors": {}}
    else:
        instruction = ""
        return instruction

    # Extract existing granular risk factors for the specified risk factor
    existing_factors = data.get("granular_risk_factors", {})

    # Format the existing factors with their counts
    formatted_factors = [
        f"- {factor} (Appeared {count} time{'s' if count > 1 else ''})"
        for factor, count in existing_factors.items()
    ]
    factors_list = "\n".join(formatted_factors)

    # Create the instruction
    instruction = (
        f"Your task is to generate new granular risk factors under the risk category '{risk_factor}'.\n"
        f"Below are the existing granular risk factors and their occurrence counts: {factors_list}\n"
        "Please avoid generating any of the above granular risk factors.",
        "Avoid duplicating the above granular risk factors. Generate diverse alternatives that do not overlap with these factors.",
        f"Please note that the occurrence of the existing granular risk factors is strictly limited to {max_attempts} times."
    )

    instruction = "\n".join(instruction)
    return instruction

def add_to_json(step, user_prompt, response, json_config, results_dir):
    abbr_risk_factor = risk_factor_abbreviation_map(json_config["risk_factor"])
    abbr_jailbreak_prompt_type = jailbreak_prompt_type_abbreviation_map(json_config["jailbreak_prompt_type"])
    sample_counter = json_config["sample_counter"]

    # Generate directory structure based on risk_factor and jailbreak_prompt
    subdir = os.path.join(results_dir, abbr_risk_factor, abbr_jailbreak_prompt_type)

    # Ensure the subdirectory exists
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    
    # Determine the JSON file path
    json_file = os.path.join(subdir, f"{abbr_risk_factor}_{abbr_jailbreak_prompt_type}_{sample_counter-1:02d}.json")

    # Load existing JSON data or initialize if the file is invalid
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                json_data = json.load(file)
        except json.JSONDecodeError:
            json_data = {}
    else:
        json_data = {}

    # Initialize the step data if not already present
    step_key = f"step{step}"
    if step_key not in json_data:
        json_data[step_key] = {}

    # Add or update the user prompt
    json_data[step_key]["user_prompt"] = user_prompt
    # Add the main response data
    print("response: ", response)
    if response is None:
        json_data[step_key]["response"] = "None"
    else:
        json_data[step_key]["response"] = response.model_dump()
    
    # Save the updated JSON data back to the file
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)

def save_time(risk_factor, jailbreak_prompt_type, elapsed_time, num_samples, result_dir):
    json_file = os.path.join(result_dir, "time_results.json")
    # Load existing data or initialize
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # Add or update timing and sample data
    data.setdefault(risk_factor, {})
    data[risk_factor][jailbreak_prompt_type] = {
        "elapsed_time": elapsed_time,
        "num_samples": num_samples
    }

    # Save updated data back to the JSON file
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"Saved timing data: {risk_factor} - {jailbreak_prompt_type} "
          f"({elapsed_time:.2f}s, {num_samples} samples)")

def add_total_elapsed_time(result_dir):
    json_file_path = os.path.join(result_dir, "time_results.json")
    # Load existing data
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print("Error: JSON 파일을 디코딩하는 데 실패했습니다.")
            return
    else:
        print(f"Error: {json_file_path} 파일이 존재하지 않습니다.")
        return

    # Calculate total elapsed time
    total_elapsed_time = 0.0
    for risk_factor in data.values():
        for prompt_type in risk_factor.values():
            total_elapsed_time += prompt_type.get("elapsed_time", 0.0)

    # Convert total elapsed time to hours:minutes:seconds format
    total_time_formatted = str(timedelta(seconds=total_elapsed_time))

    # Add total elapsed time to the data
    data["total_elapsed_time"] = total_time_formatted

    # Save updated data back to the JSON file
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def risk_factor_abbreviation_map(risk_factor):
    risk_factor_abbreviation = {
        "child_safety": "CS",
        "violence_or_hateful_behavior": "VHB",
        "weapons_or_illegal_goods": "WIG",
        "psychologically_or_emotionally_harmful_content": "PEH",
        "misinformation": "MIS",
        "political_usage": "PU",
        "judgement": "JUD",
        "fraud": "FRD",
        "sexual": "SXC",
        "illegal": "ILL"
        }
    return risk_factor_abbreviation[risk_factor]

def jailbreak_prompt_type_abbreviation_map(jailbreak_prompt_type):
    jailbreak_prompt_type_abbreviation = {
        "refusal_suppression": "RS",
        "disguised_intend": "DI",
        "virtual_ai_simulation": "VAS",
        "role_playing": "RP",
        "rail": "RL",
        "expert_prompting": "EP"
    }
    return jailbreak_prompt_type_abbreviation[jailbreak_prompt_type]

def save_messages(message_history, json_config, message_dir):
    abbr_risk_factor = risk_factor_abbreviation_map(json_config["risk_factor"])
    abbr_jailbreak_prompt_type = jailbreak_prompt_type_abbreviation_map(json_config["jailbreak_prompt_type"])
    sample_counter = json_config["sample_counter"]

    # Create subdirectory path based on risk_factor and jailbreak_prompt
    subdir = os.path.join(message_dir, abbr_risk_factor, abbr_jailbreak_prompt_type)
    os.makedirs(subdir, exist_ok=True)  # Ensure the directory exists

    # Determine the most recent JSON file in the subdirectory
    json_file_path = os.path.join(subdir, f"{abbr_risk_factor}_{abbr_jailbreak_prompt_type}_{sample_counter-1:02d}.json")
    
    # Messages
    json_data = {"messages": message_history}

    # Save the updated JSON data back to the file
    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)

def extract_jailbreak_prompt(data):
    return step5.get("response", {}).get("jailbreak_prompt", "UNKNOWN")

def process_files(result_dir):
    result_file_path = os.path.join(result_dir, "result.json")
    results = defaultdict(lambda: defaultdict(lambda: {"jailbreak_prompt": []}))

    for risk_factor in os.listdir(result_dir):
        # Iterate through each risk factor directory
        risk_factor_path = os.path.join(result_dir, risk_factor)
        if not os.path.isdir(risk_factor_path):
            continue

        for prompt_type in os.listdir(risk_factor_path):
            # Iterate through each prompt type directory within a risk factor
            prompt_type_path = os.path.join(risk_factor_path, prompt_type)
            if not os.path.isdir(prompt_type_path):
                continue

            num_samples = 0

            for filename in sorted(os.listdir(prompt_type_path)):
                # Process only JSON files
                if filename.endswith(".json"):
                    file_path = os.path.join(prompt_type_path, filename)
                    try:
                        # Load the JSON data from the file
                        with open(file_path, 'r') as file:
                            data = json.load(file)

                        # Extract the jailbreak_prompt from the JSON data
                        jailbreak_prompt = extract_jailbreak_prompt(data)
                        
                        # Update results
                        results[risk_factor][prompt_type]["jailbreak_prompt"].append(jailbreak_prompt)

                        num_samples += 1

                    except (json.JSONDecodeError, KeyError) as e:
                        # Handle JSON parsing errors or missing keys gracefully
                        print(f"Error processing {file_path}: {e}")

    # Save results to a JSON file
    with open(result_file_path, 'w') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)