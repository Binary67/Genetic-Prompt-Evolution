import pandas as pd
from GPEvolution import RunEvolution

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

    BestPrompt = RunEvolution(
        TrainingData = TrainingData,
        ValidationData = ValidationData,
        ClassificationLabels = ["compliment", "development"],
        PopulationSize = 2,
        NumGenerations = 2,
        MutationRate = 0.3,
        EliteSize = 2,
        RoleAssignment = "You are an expert HR analyst specializing in employee feedback classification."
    )