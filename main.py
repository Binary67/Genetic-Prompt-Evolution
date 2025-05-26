import pandas as pd
from GPEvolution import RunEvolution
from GPEvaluate import EvaluateValidationWithBestPrompt
from sklearn.model_selection import train_test_split
import numpy as np

# TrainingData = pd.DataFrame([
#     {"text": "Great job on the project presentation", "label": "compliment"},
#     {"text": "Your code quality is excellent", "label": "compliment"},
#     {"text": "Needs improvement in communication skills", "label": "development"},
#     {"text": "Should work on time management", "label": "development"},
#     {"text": "Meeting attendance has been consistent", "label": "neutral"},
#     {"text": "Submitted all required documentation on time", "label": "neutral"},
#     {"text": "Follows standard procedures correctly", "label": "neutral"},
#     {"text": "Maintains professional demeanor at work", "label": "neutral"},
# ])

# ValidationData = pd.DataFrame([
#     {"text": "Excellent teamwork and collaboration", "label": "compliment"},
#     {"text": "Need to improve documentation practices", "label": "development"},
#     {"text": "Your innovative approach is commendable", "label": "compliment"},
#     {"text": "Should focus on technical skill development", "label": "development"},
#     {"text": "Completes assigned tasks as expected", "label": "neutral"},
#     {"text": "Participates in team meetings regularly", "label": "neutral"},
#     {"text": "Work output meets basic requirements", "label": "neutral"},
#     {"text": "Adheres to company policies", "label": "neutral"},
# ])

DataTA = pd.read_excel('your_file.xlsx')
DataTA = DataTA.dropna(subset = ['Validation'])
 
DataTA['GroundTruth'] = np.where(
    DataTA['Validation'] == 'Agree',
    DataTA['has_aspiration'],
    np.where(DataTA['has_aspiration'] == 'Yes', 'No', 'Yes')
)
 
DataTA = DataTA.rename(columns = {'GroundTruth': 'label', 'talent_statement': 'text'})
DataTA = DataTA[['text', 'label']]
DataTA = DataTA.replace({'label': {'Yes': 'has_aspiration', 'No': 'no_aspiration'}})
 
TrainingData, ValidationData = train_test_split(DataTA, test_size = 0.33, stratify = DataTA['label'])
Labels = TrainingData['label'].unique().tolist()

if __name__ == "__main__":

    BestPrompt = RunEvolution(
        InputData = TrainingData,
        ClassificationLabels = Labels,
        PopulationSize = 15,
        NumGenerations = 10,
        MutationRate = 0.3,
        EliteSize = 2,
        RoleAssignment = "You are an expert HR analyst specializing in identifying employee aspirations based on feedback statements. If the input text is unclear or ambiguous, you should classify it as 'no_aspiration'. "
    )
    
    # Evaluate validation data with the best prompt
    ValidationResults = EvaluateValidationWithBestPrompt(
        ValidationData = ValidationData,
        BestPrompt = BestPrompt,
        ClassificationLabels = Labels,
    )