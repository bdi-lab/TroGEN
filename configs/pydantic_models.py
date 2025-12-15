from pydantic import BaseModel

class RiskSelection(BaseModel):
    granular_risk_factor: str

class EntityCreation(BaseModel):
    subject: str

class ScenarioDesign(BaseModel):
    scenario: str

class PromptGeneration(BaseModel):
    prompt: str

class JailbreakImplementation(BaseModel):
    jailbreak_prompt: str

class Examination(BaseModel):
    answer: str
    explanation: str