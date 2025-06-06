import random
from typing import Dict, List
from PromptStructure import GeneratePromptSample, ParsePromptWithDelimiter, CombineString
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI client
Client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

def Crossover(Parent1: Dict, Parent2: Dict) -> Dict:
    """
    Perform crossover between two parent prompts to create offspring.
    
    Parameters:
    - Parent1: First parent prompt dictionary
    - Parent2: Second parent prompt dictionary
    
    Returns:
    - Offspring prompt dictionary
    """
    # Create offspring dictionary
    Offspring = {}
    
    # List of all prompt component keys
    PromptKeys = [
        "RoleAssignment",
        "PerspectiveSetting",
        "ContextInfo",
        "TaskInstruction",
        "LabelSetDefinition",
        "OutputFormat",
        "ReasoningDirective",
        "FewShotBlock"
    ]
    
    # Randomly select components from each parent
    for Key in PromptKeys:
        # 50% chance to inherit from Parent1, 50% from Parent2
        if random.random() < 0.5:
            Offspring[Key] = Parent1[Key]
        else:
            Offspring[Key] = Parent2[Key]
    
    return Offspring

def Mutation(Individual: Dict, MutationRate: float = 0.3, ClassificationLabels: List[str] = None) -> Dict:
    """
    Perform mutation on an individual prompt.
    
    Parameters:
    - Individual: Prompt dictionary to mutate
    - MutationRate: Probability of mutating each component (default 0.3)
    - ClassificationLabels: List of classification labels for context
    
    Returns:
    - Mutated prompt dictionary
    """
    # Create a copy of the individual
    Mutated = Individual.copy()
    
    # List of all prompt component keys
    PromptKeys = [
        "RoleAssignment",
        "PerspectiveSetting",
        "ContextInfo",
        "TaskInstruction",
        "LabelSetDefinition",
        "OutputFormat",
        "ReasoningDirective",
        "FewShotBlock"
    ]
    
    # For each component, decide whether to mutate
    for Key in PromptKeys:
        if random.random() < MutationRate:
            # Mutate this component by generating a new version
            Mutated[Key] = MutateComponent(Key, Individual[Key], ClassificationLabels)
    
    return Mutated

def MutateComponent(ComponentName: str, OriginalValue: str, ClassificationLabels: List[str] = None) -> str:
    """
    Mutate a specific prompt component using LLM.
    
    Parameters:
    - ComponentName: Name of the component to mutate
    - OriginalValue: Original value of the component
    - ClassificationLabels: List of classification labels for context
    
    Returns:
    - Mutated component string
    """
    SystemMessage = {
        "role": "system",
        "content": "You are a prompt engineering expert specializing in creating variations of prompt components."
    }
    
    UserMessage = {
        "role": "user",
        "content": (
            f"Create a variation of the following prompt component.\n"
            f"Component Type: {ComponentName}\n"
            f"Original:\n{OriginalValue}\n\n"
            f"Create a different version that maintains the same purpose but uses different wording, "
            f"structure, or approach. The variation should be meaningful and not just a minor rephrasing.\n"
            + (f"Context: This is for a classification task with labels: {', '.join(ClassificationLabels)}\n" if ClassificationLabels else "")
            + f"Output ONLY the new component text, nothing else."
        )
    }
    
    try:
        Response = Client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[SystemMessage, UserMessage],
            temperature=0.8,
            max_tokens=500
        )
        
        MutatedValue = Response.choices[0].message.content.strip()
        return MutatedValue
    except Exception as e:
        print(f"Mutation failed for {ComponentName}: {str(e)}")
        # Return original value if mutation fails
        return OriginalValue

def TournamentSelection(Population: List[Dict], FitnessScores: List[float], TournamentSize: int = 3) -> Dict:
    """
    Select an individual from the population using tournament selection.
    
    Parameters:
    - Population: List of prompt dictionaries
    - FitnessScores: List of fitness scores corresponding to each individual
    - TournamentSize: Number of individuals to compete in each tournament
    
    Returns:
    - Selected individual
    """
    # Randomly select tournament participants
    TournamentIndices = random.sample(range(len(Population)), min(TournamentSize, len(Population)))
    
    # Find the best individual in the tournament
    BestIndex = TournamentIndices[0]
    BestFitness = FitnessScores[TournamentIndices[0]]
    
    for Index in TournamentIndices[1:]:
        if FitnessScores[Index] > BestFitness:
            BestFitness = FitnessScores[Index]
            BestIndex = Index
    
    return Population[BestIndex]