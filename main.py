import pandas as pd
from GPEvolution import RunEvolution
from GPEvaluate import EvaluateValidationWithBestPrompt

TrainingData = pd.DataFrame([
    {"text": "Great job on the project presentation", "label": "compliment"},
    {"text": "Your code quality is excellent", "label": "compliment"},
    {"text": "Needs improvement in communication skills", "label": "development"},
    {"text": "Should work on time management", "label": "development"},
    {"text": "Meeting attendance has been consistent", "label": "neutral"},
    {"text": "Submitted all required documentation on time", "label": "neutral"},
    {"text": "Follows standard procedures correctly", "label": "neutral"},
    {"text": "Maintains professional demeanor at work", "label": "neutral"},
])

ValidationData = pd.DataFrame([
    {"text": "Excellent teamwork and collaboration", "label": "compliment"},
    {"text": "Need to improve documentation practices", "label": "development"},
    {"text": "Your innovative approach is commendable", "label": "compliment"},
    {"text": "Should focus on technical skill development", "label": "development"},
    {"text": "Completes assigned tasks as expected", "label": "neutral"},
    {"text": "Participates in team meetings regularly", "label": "neutral"},
    {"text": "Work output meets basic requirements", "label": "neutral"},
    {"text": "Adheres to company policies", "label": "neutral"},
])

if __name__ == "__main__":

    BestPrompt = RunEvolution(
        InputData = TrainingData,
        ClassificationLabels = ["compliment", "development", "neutral"],
        PopulationSize = 2,
        NumGenerations = 2,
        MutationRate = 0.3,
        EliteSize = 2,
        RoleAssignment = "You are an expert HR analyst specializing in employee feedback classification."
    )
    
    # Evaluate validation data with the best prompt
    ValidationResults = EvaluateValidationWithBestPrompt(
        ValidationData = ValidationData,
        BestPrompt = BestPrompt,
        ClassificationLabels = ["compliment", "development", "neutral"]
    )