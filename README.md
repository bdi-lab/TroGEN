# Beneath the Facade: Probing Safety Vulnerabilities in LLMs via Auto-Generated Jailbreak Prompts

This is the official implementation of the paper:

**Beneath the Facade: Probing Safety Vulnerabilities in LLMs via Auto-Generated Jailbreak Prompts**  
*To appear in Findings of the Association for Computational Linguistics: EMNLP 2025*

All code was written by **Heehyeon Kim** (heehyeon@kaist.ac.kr) and **Kyeongryul Lee** (klee0257@kaist.ac.kr).  
If you use this repository, please cite our paper.

## Requirements

- Python 3.9.18  
- openai 1.59.4  

Install all dependencies (except Python) with:

```bash
pip install -r requirements.txt
```


## Directory Structure

```
./configs/info.json
    ├── Defines the experimental process, including:
    │   • List of risk factors
    │   • Definitions of each risk factor
    │   • System and user prompt templates for each pipeline step
    │   • Jailbreak prompt types (with definitions and examples)

./dataset/{DATA_TYPE}/jailbreak_prompts
    └── Final jailbreak prompts generated for each risk factor × jailbreak type

./dataset/{DATA_TYPE}/storage
    └── Intermediate storage for granular risk factors (prevents recomputation)

./dataset/{DATA_TYPE}/message_history
    └── Full conversation histories (system + user prompts, model responses) for debugging and reproducibility
```


## Key Parameters

- **`DATA_TYPE`**  
  Defines which dataset split is processed (e.g., `"original"`, `"augmented"`).  
  Used to separate result directories.

- **`risk_factors` (from config)**  
  Codes for high-level risk categories (e.g., `FRD`, `PU`, `ILL`, `SXC`).  
  Each risk factor is processed through the full pipeline.

- **`jailbreak_prompts` (from config)**  
  Dictionary of jailbreak strategies per risk factor:  
  - *Definition*: Definitions of jailbreak prompting strategies for an adversarial agent
  - *Examples*: PExamples of jailbreak prompt types for each risk factor

- **`num_samples`**  
  Number of independent samples generated per `(risk factor × jailbreak type)` pair.


## Jailbreak Prompt Generation

We provide the checkpoints used to reproduce all reported results.  
Run the pipeline with:

```bash
python main.py
```

---

## License

This code is released under the **CC BY-NC-SA 4.0 license**.  
