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

def EvaluatePrompt(DataFrame: pd.DataFrame, Prompt: Dict) -> pd.DataFrame:
    """
    Evaluate a prompt on a dataframe using Azure OpenAI.
    
    Args:
        DataFrame: Input dataframe containing data to classify
        Prompt: Prompt dictionary from PromptStructure.py
        
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
        CurrentPrompt = CurrentPrompt + "\n\nText to classify: " + TextToClassify
        
        # Create messages for Azure OpenAI
        Messages = [
            {"role": "user", "content": CurrentPrompt}
        ]
        
        try:
            # Call Azure OpenAI
            Response = Client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=Messages,
                temperature=0,  # Use 0 for consistent classification
                max_tokens=100  # Limit response length for classification
            )
            
            # Extract prediction from response
            RawPrediction = Response.choices[0].message.content.strip()
            
            # Search for the first occurrence of 'compliment' or 'development' (case-insensitive)
            Match = re.search(r'\b(compliment|development)\b', RawPrediction, re.IGNORECASE)
            
            if Match:
                Prediction = Match.group(1).lower()  # Extract and lowercase the label
            else:
                # If no match found, default to error
                Prediction = "ERROR: No label found"
                print(f"Warning: Could not extract label from response: {RawPrediction[:100]}...")
                
            Predictions.append(Prediction)
            RawOutput.append(RawPrediction)
            
        except Exception as E:
            # Handle errors gracefully
            print(f"Error processing row {Index}: {str(E)}")
            Predictions.append("ERROR")
    
    # Add predictions to the dataframe
    ResultDF['prediction'] = Predictions
    ResultDF['raw_output'] = RawOutput
    
    return ResultDF