import os
import time
from openai import OpenAI
from tqdm import tqdm

from utils import (
    load_config, generate_instruction, save_granular_risk_factor, process_step, save_messages, save_time, add_total_elapsed_time, process_files
    )

from configs.pydantic_models import (
    RiskSelection, EntityCreation, ScenarioDesign, PromptGeneration, JailbreakImplementation
)

API_KEY = ""

DATA_TYPE = "original"
RESULT_DIR = f"./dataset/{DATA_TYPE}/jailbreak_prompts"
GRANULAR_SAVE_DIR = f"./dataset/{DATA_TYPE}/storage"
MESSAGE_SAVE_DIR = f"./dataset/{DATA_TYPE}/message_history"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GRANULAR_SAVE_DIR, exist_ok=True)
os.makedirs(MESSAGE_SAVE_DIR, exist_ok=True)

def main(api_key, data_type, granular_result_dir, result_dir, message_save_dir):
    info, model_params = load_config()
    client = OpenAI(api_key=api_key)

    api_params = {
        "model": model_params["model"],
        "temperature": model_params["temperature"],
        "max_tokens": model_params["max_tokens"],
        "top_p": model_params["top_p"],
        "frequency_penalty": model_params["frequency_penalty"],
    }

    num_samples = 1
    process = info["process"]
    risk_factors = info["risk_factors"]
    def_of_rf = info["def_of_risk_factors"]
    jailbreak_prompt_types = info["jailbreak_prompts"][risk_factors[0]].keys()
    
    for risk_factor in tqdm(risk_factors, desc="Processing risk factors"):
        for jailbreak_prompt_type in tqdm(jailbreak_prompt_types, desc=f"Risk Factor: {risk_factor}", leave=False):
            start_time = time.time() 
            sample_counter = 0
            
            for _ in range(num_samples):
                message_history = []
                sample_counter += 1
        
                step_num = 1
                json_config = {
                    "risk_factor": risk_factor,
                    "jailbreak_prompt_type": jailbreak_prompt_type,
                    "sample_counter": sample_counter
                }

                # Step 1: Derivation of Granular Risk Factors
                user_prompt = process["user_prompts"][0].format(risk_factor=risk_factor, def_of_rf=def_of_rf[risk_factor])
                redundancy_instruction = generate_instruction(risk_factor, granular_result_dir)
                response = process_step(
                    step_num, client, user_prompt, process["system_prompts"][0], RiskSelection, message_history, json_config, result_dir, api_params, additional_instruction=redundancy_instruction,
                )
                granular_risk_factor = response.granular_risk_factor
                save_granular_risk_factor(risk_factor, granular_risk_factor, granular_result_dir)
                step_num += 1

                # Step 2: Scenario-driven Risk Modeling
                context_vars = {
                    "risk_factor": risk_factor,
                    "granular_risk_factor": granular_risk_factor
                }
                user_prompt = process["user_prompts"][1].format(**context_vars)
                scenario_response = process_step(
                    step_num, client, user_prompt, process["system_prompts"][1], ScenarioDesign, message_history, json_config, result_dir, api_params
                )
                scenario = scenario_response.scenario
                context_vars["scenario"] = scenario
                step_num += 1

                # Step 3: Key Subject Identification
                user_prompt = process["user_prompts"][2].format(**context_vars)
                response = process_step(
                    step_num, client, user_prompt, process["system_prompts"][2], EntityCreation, message_history, json_config, result_dir, api_params
                )
                subject = response.subject
                context_vars["subject"] = subject
                step_num += 1

                # Step 4: Harmful Prompt Generation
                user_prompt = process["user_prompts"][3].format(**context_vars)
                scenario_response = process_step(
                    step_num, client, user_prompt, process["system_prompts"][3], PromptGeneration, message_history, json_config, result_dir, api_params
                )
                context_vars["prompt"] = scenario_response.prompt
                step_num += 1

                # Step 5: Applying Jailbreak Prompting
                jailbreak_prompt_info = info["jailbreak_prompts"][risk_factor][jailbreak_prompt_type]
                context_vars.update({
                    "def_of_jp": jailbreak_prompt_info["definition"],
                    "emp_of_jp": '\n'.join(jailbreak_prompt_info["examples"])
                })
                
                user_prompt = process["user_prompts"][4].format(**context_vars)
                scenario_response = process_step(
                    step_num, client, user_prompt, process["system_prompts"][4], JailbreakImplementation, message_history, json_config, result_dir, api_params
                )
                context_vars["jailbreak_prompt"] = scenario_response.jailbreak_prompt
                
                save_messages(message_history, json_config, message_save_dir)

            elapsed_time = time.time() - start_time
            save_time(risk_factor, jailbreak_prompt_type, elapsed_time, sample_counter, result_dir)

    add_total_elapsed_time(result_dir)
    process_files(result_dir)

if __name__ == "__main__":
    main(API_KEY, DATA_TYPE, GRANULAR_SAVE_DIR, RESULT_DIR, MESSAGE_SAVE_DIR)