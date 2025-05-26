import pandas as pd
from typing import Dict
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from PromptStructure import CombineString
import re

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI client
Client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

def EvaluatePrompt(DataFrame: pd.DataFrame, Prompt: Dict, ClassificationLabels: list = None) -> pd.DataFrame:
    """
    Evaluate a prompt on a dataframe using Azure OpenAI.
    
    Args:
        DataFrame: Input dataframe containing data to classify
        Prompt: Prompt dictionary from PromptStructure.py
        ClassificationLabels: List of valid classification labels
        
    Returns:
        DataFrame with additional 'prediction' column
    """
    # Convert prompt dictionary to string if needed
    if isinstance(Prompt, dict):
        PromptString = CombineString(Prompt)
    else:
        PromptString = Prompt
    
    # Create a copy of the dataframe to avoid modifying the original
    ResultDF = DataFrame.copy()
    
    # Initialize predictions list
    Predictions = []
    RawOutput = []
    
    # Process each row in the dataframe
    for Index, Row in DataFrame.iterrows():
        # Since InputText is no longer in the prompt, we always append the text
        CurrentPrompt = PromptString
        
        # Get the text to classify from the 'text' column
        if 'text' in Row:
            TextToClassify = Row['text']
        else:
            # Fallback to string representation of the entire row
            TextToClassify = str(Row.to_dict())
        
        # Append the text to classify at the end of the prompt
        CurrentPrompt = CurrentPrompt + "\n\n<INPUT_TEXT>Text to classify: " + TextToClassify + "</INPUT_TEXT>"
        
        # Create messages for Azure OpenAI
        Messages = [
            {"role": "user", "content": CurrentPrompt}
        ]
        
        # Retry logic - up to 25 attempts
        MaxRetries = 25
        RetryCount = 0
        Success = False
        
        while RetryCount < MaxRetries and not Success:
            try:
                # Prepare messages for current attempt - create a fresh copy each time
                CurrentMessages = [{"role": "user", "content": Messages[0]["content"]}]
                
                # If this is a retry, add emphasis message about using only given labels
                if RetryCount > 0:
                    EmphasisMessage = f"\nIMPORTANT: Your output MUST be ONLY one of these exact labels: {', '.join(ClassificationLabels)}. Do not use any other words or labels. If the input text is unclear or ambiguous, classify it as 'no_aspiration'."
                    CurrentMessages[0]["content"] += EmphasisMessage
                
                # Call Azure OpenAI
                Response = Client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    messages=CurrentMessages,
                    temperature=0.5,  # Use 0 for consistent classification
                )
                
                # Extract prediction from response
                RawPrediction = Response.choices[0].message.content.strip()
                
                # Build regex pattern dynamically from ClassificationLabels
                LabelPattern = '|'.join(re.escape(label) for label in ClassificationLabels)
                Pattern = rf'\b({LabelPattern})\b'
                    
                # Search for the first occurrence of any classification label (case-insensitive)
                Match = re.search(Pattern, RawPrediction, re.IGNORECASE)
                
                if Match:
                    Prediction = Match.group(1).lower()  # Extract and lowercase the label
                    Success = True
                else:
                    # If no match found, retry
                    RetryCount += 1
                    if RetryCount < MaxRetries:
                        print(f"Warning: Could not extract label from response (attempt {RetryCount}): {RawPrediction}")
                    else:
                        # Max retries reached
                        Prediction = "ERROR: No label found after 25 attempts"
                        print(f"Error: Could not extract label after {MaxRetries} attempts. Last response: {RawPrediction}")
                        Success = True  # Exit loop
                
            except Exception as E:
                # Handle API errors
                RetryCount += 1
                if RetryCount < MaxRetries:
                    print(f"Error processing row {Index} (attempt {RetryCount}): {str(E)}")
                else:
                    # Max retries reached
                    print(f"Error processing row {Index} after {MaxRetries} attempts: {str(E)}")
                    Prediction = "ERROR"
                    RawPrediction = f"API Error: {str(E)}"
                    Success = True  # Exit loop
        
        Predictions.append(Prediction)
        RawOutput.append(RawPrediction)
    
    # Add predictions to the dataframe
    ResultDF['prediction'] = Predictions
    ResultDF['raw_output'] = RawOutput
    
    return ResultDF

def CalculateFitnessScore(ResultDF: pd.DataFrame) -> float:
    """
    Calculate fitness score (accuracy) from evaluation results.
    
    Args:
        ResultDF: DataFrame containing 'label' and 'prediction' columns
        
    Returns:
        float: Accuracy percentage (0-100)
    """
    Correct = (ResultDF['label'] == ResultDF['prediction']).sum()
    Total = len(ResultDF)
    Accuracy = Correct / Total * 100
    
    return Accuracy

def GetWrongPredictions(ResultDF: pd.DataFrame) -> pd.DataFrame:
    """
    Extract wrong predictions from evaluation results.
    
    Args:
        ResultDF: DataFrame containing evaluation results
        
    Returns:
        DataFrame containing only misclassified examples
    """
    WrongPredictions = ResultDF[ResultDF['label'] != ResultDF['prediction']].copy()
    return WrongPredictions

def EvaluateValidationWithBestPrompt(ValidationData: pd.DataFrame, BestPrompt: Dict, ClassificationLabels: list = None) -> Dict:
    """
    Evaluate validation data using the best prompt from final generation.
    
    Args:
        ValidationData: DataFrame containing validation data with 'text' and 'label' columns
        BestPrompt: Best prompt dictionary from evolution
        ClassificationLabels: List of valid classification labels
        
    Returns:
        Dict containing evaluation results and metrics
    """
    print("\n" + "="*50)
    print("VALIDATION EVALUATION")
    print("="*50)
    
    # Evaluate validation data with best prompt
    ResultDF = EvaluatePrompt(ValidationData, BestPrompt, ClassificationLabels)
    
    # Calculate validation accuracy
    ValidationAccuracy = CalculateFitnessScore(ResultDF)
    
    # Calculate per-class accuracy
    ClassAccuracies = {}
    for Label in ValidationData['label'].unique():
        LabelMask = ValidationData['label'] == Label
        LabelCorrect = (ResultDF.loc[LabelMask, 'label'] == ResultDF.loc[LabelMask, 'prediction']).sum()
        LabelTotal = LabelMask.sum()
        ClassAccuracies[Label] = (LabelCorrect / LabelTotal * 100) if LabelTotal > 0 else 0
    
    # Prepare results summary
    Results = {
        "OverallAccuracy": ValidationAccuracy,
        "ClassAccuracies": ClassAccuracies,
        "TotalSamples": len(ValidationData),
        "CorrectPredictions": (ResultDF['label'] == ResultDF['prediction']).sum(),
        "IncorrectPredictions": (ResultDF['label'] != ResultDF['prediction']).sum(),
        "ResultDataFrame": ResultDF
    }
    
    # Print validation results
    print(f"\nValidation Results:")
    print(f"Overall Accuracy: {ValidationAccuracy:.2f}%")
    print(f"Total Samples: {Results['TotalSamples']}")
    print(f"Correct Predictions: {Results['CorrectPredictions']}")
    print(f"Incorrect Predictions: {Results['IncorrectPredictions']}")
    
    print(f"\n### Per-Class Accuracy: ###")
    for Label, Accuracy in ClassAccuracies.items():
        print(f"{Label}: {Accuracy:.2f}%")
    
    # Show misclassified examples
    Misclassified = ResultDF[ResultDF['label'] != ResultDF['prediction']]
    if len(Misclassified) > 0:
        print(f"\nMisclassified Examples:")
        for Index, Row in Misclassified.iterrows():
            print(f"\nText: {Row['text']}")
            print(f"Actual: {Row['label']}, Predicted: {Row['prediction']}")
    
    return Results