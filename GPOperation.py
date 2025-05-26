import random
from typing import Dict, List
from PromptStructure import GeneratePromptSample, ParsePromptWithDelimiter, CombineString
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import pandas as pd

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

def Mutation(Individual: Dict, MutationRate: float = 0.3, ClassificationLabels: List[str] = None, WrongPredictions: pd.DataFrame = None) -> Dict:
    """
    Perform mutation on an individual prompt.
    
    Parameters:
    - Individual: Prompt dictionary to mutate
    - MutationRate: Probability of mutating each component (default 0.3)
    - ClassificationLabels: List of classification labels for context
    - WrongPredictions: DataFrame containing misclassified examples
    
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
            Mutated[Key] = MutateComponent(Key, Individual[Key], ClassificationLabels, WrongPredictions)
    
    return Mutated

def MutateComponent(ComponentName: str, OriginalValue: str, ClassificationLabels: List[str] = None, WrongPredictions: pd.DataFrame = None) -> str:
    """
    Mutate a specific prompt component using LLM with error analysis and targeted fixes.
    
    Parameters:
    - ComponentName: Name of the component to mutate
    - OriginalValue: Original value of the component
    - ClassificationLabels: List of classification labels for context
    - WrongPredictions: DataFrame containing misclassified examples
    
    Returns:
    - Mutated component string
    """
    # If we have wrong predictions, first analyze why they failed
    ErrorAnalysis = ""
    if WrongPredictions is not None and len(WrongPredictions) > 0:
        ErrorAnalysis = AnalyzeClassificationErrors(WrongPredictions, OriginalValue, ComponentName, ClassificationLabels)
    
    SystemMessage = {
        "role": "system",
        "content": "You are a prompt engineering expert specializing in creating variations of prompt components that fix classification errors based on error analysis."
    }
    
    # Build error context if wrong predictions are provided
    ErrorContext = ""
    if WrongPredictions is not None and len(WrongPredictions) > 0:
        # Sample up to 5 wrong predictions for context
        SampleErrors = WrongPredictions.sample(min(5, len(WrongPredictions)))
        ErrorExamples = []
        for _, Row in SampleErrors.iterrows():
            ErrorExamples.append(f"- Text: \"{Row['text']}\"\n  Actual: {Row['label']}, Predicted: {Row['prediction']}")
        
        ErrorContext = (
            f"\n\nIMPORTANT: The current prompt misclassified these examples:\n"
            + "\n".join(ErrorExamples)
            + f"\n\nError Analysis:\n{ErrorAnalysis}\n"
            + f"\n\nThe mutation should specifically address the identified issues."
        )
    
    UserMessage = {
        "role": "user",
        "content": (
            f"Create an improved variation of the following prompt component based on the error analysis.\n"
            f"Component Type: {ComponentName}\n"
            f"Original:\n{OriginalValue}\n\n"
            f"Create a different version that specifically addresses the identified issues while maintaining the same purpose.\n"
            f"Focus on fixing the root causes of misclassifications identified in the error analysis.\n"
            + (f"Context: This is for a classification task with labels: {', '.join(ClassificationLabels)}\n" if ClassificationLabels else "")
            + ErrorContext
            + f"\n\nOutput ONLY the new component text, nothing else."
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

def AnalyzeClassificationErrors(WrongPredictions: pd.DataFrame, OriginalComponent: str, ComponentName: str, ClassificationLabels: List[str] = None) -> str:
    """
    Analyze why classifications failed to guide targeted mutations.
    
    Parameters:
    - WrongPredictions: DataFrame containing misclassified examples
    - OriginalComponent: The original component value
    - ComponentName: Name of the component being analyzed
    - ClassificationLabels: List of classification labels for context
    
    Returns:
    - Analysis of error patterns and suggested improvements
    """
    SystemMessage = {
        "role": "system",
        "content": "You are an expert at analyzing classification errors and identifying patterns in misclassifications."
    }
    
    # Sample errors for analysis
    SampleErrors = WrongPredictions.sample(min(10, len(WrongPredictions)))
    ErrorExamples = []
    for _, Row in SampleErrors.iterrows():
        ErrorExamples.append(f"- Text: \"{Row['text']}\"\n  Actual: {Row['label']}, Predicted: {Row['prediction']}")
    
    UserMessage = {
        "role": "user",
        "content": (
            f"Analyze these classification errors and identify why they might have occurred:\n\n"
            f"Component being analyzed: {ComponentName}\n"
            f"Current component value:\n{OriginalComponent}\n\n"
            f"Misclassified examples:\n"
            + "\n".join(ErrorExamples)
            + (f"\n\nAvailable labels: {', '.join(ClassificationLabels)}\n" if ClassificationLabels else "")
            + f"\n\nProvide a concise analysis of:\n"
            f"1. Common patterns in the misclassifications\n"
            f"2. Potential weaknesses in the current component that led to these errors\n"
            f"3. Specific improvements that could address these issues\n\n"
            f"Be specific and actionable in your analysis."
        )
    }
    
    try:
        Response = Client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[SystemMessage, UserMessage],
            temperature=0.3,
            max_tokens=300
        )
        
        return Response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error analysis failed: {str(e)}")
        return "Error analysis unavailable"

def TournamentSelection(Population: List[Dict], FitnessScores: List[float], TournamentSize: int = 3, ReturnIndex: bool = False):
    """
    Select an individual from the population using tournament selection.
    
    Parameters:
    - Population: List of prompt dictionaries
    - FitnessScores: List of fitness scores corresponding to each individual
    - TournamentSize: Number of individuals to compete in each tournament
    - ReturnIndex: If True, return the index instead of the individual
    
    Returns:
    - Selected individual or index
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
    
    if ReturnIndex:
        return BestIndex
    return Population[BestIndex]