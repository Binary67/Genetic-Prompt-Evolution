from typing import List, Dict
from PromptStructure import GeneratePromptSample

def GeneratePopulation(PopulationSize: int, 
                      RoleAssignment: str = "You are an expert in classification.", 
                      ClassificationLabels: List[str] = None) -> List[Dict]:
    """
    Generate a population of prompts for genetic programming.
    
    Parameters:
    - PopulationSize: Number of prompts to generate
    - RoleAssignment: The role to assign to the model
    - ClassificationLabels: List of classification labels
    
    Returns:
    - List of prompt dictionaries
    """
    Population = []
    
    for i in range(PopulationSize):
        PromptDict = GeneratePromptSample(
            RoleAssignment=RoleAssignment,
            ClassificationLabels=ClassificationLabels
        )
        Population.append(PromptDict)
    
    return Population