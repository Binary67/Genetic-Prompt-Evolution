import pandas as pd
from PromptStructure import GeneratePromptSample, CombineString
from GPEvaluate import EvaluatePrompt
from GPPopulation import GeneratePopulation

TrainingData = pd.DataFrame([
    {"text": "Great job on the project presentation", "label": "compliment"},
    {"text": "Your code quality is excellent", "label": "compliment"},
    {"text": "Needs improvement in communication skills", "label": "development"},
    {"text": "Should work on time management", "label": "development"},
])

ValidationData = pd.DataFrame([
    {"text": "Excellent teamwork and collaboration", "label": "compliment"},
    {"text": "Need to improve documentation practices", "label": "development"},
    {"text": "Your innovative approach is commendable", "label": "compliment"},
    {"text": "Should focus on technical skill development", "label": "development"},
])

if __name__ == "__main__":
    # Generate a population of prompts for classification
    ClassificationLabels = ["compliment", "development"]
    PopulationSize = 5
    
    print(f"Generating population of {PopulationSize} prompts...")
    PromptPopulation = GeneratePopulation(
        PopulationSize=PopulationSize,
        RoleAssignment="You are an expert HR analyst specializing in employee feedback classification.",
        ClassificationLabels=ClassificationLabels
    )
    
    # Evaluate each prompt in the population on validation data
    print(f"\n\nEvaluating {PopulationSize} prompts on validation data...")
    
    BestAccuracy = 0
    BestPromptIndex = 0
    
    for i, PromptDict in enumerate(PromptPopulation):
        print(f"\nEvaluating Prompt {i+1}...")
        ResultDF = EvaluatePrompt(ValidationData, PromptDict)
        
        # Calculate accuracy
        Correct = (ResultDF['label'] == ResultDF['prediction']).sum()
        Total = len(ResultDF)
        Accuracy = Correct / Total * 100
        
        print(f"Prompt {i+1} Accuracy: {Accuracy:.2f}% ({Correct}/{Total})")
        
        if Accuracy > BestAccuracy:
            BestAccuracy = Accuracy
            BestPromptIndex = i
    
    print(f"\n\nBest Prompt: Prompt {BestPromptIndex+1} with {BestAccuracy:.2f}% accuracy")