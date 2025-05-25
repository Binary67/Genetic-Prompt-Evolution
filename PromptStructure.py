from typing import Dict, List
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

def GeneratePromptSample(RoleAssignment: str = "You are an expert in classification.", 
                         ClassificationLabels: List[str] = None) -> dict:

    PromptSkeleton = """{RoleAssignment}
{Delimiter}
{PerspectiveSetting}
{Delimiter}
{ContextInfo}
{Delimiter}
{TaskInstruction}
{Delimiter}
{LabelSetDefinition}
{Delimiter}
{OutputFormat}
{Delimiter}
{ReasoningDirective}
{Delimiter}
{FewShotBlock}
{Delimiter}
"""

    SystemMessage = {
        "role": "system",
        "content": "You are a top-tier prompt-engineering assistant."
    }

    UserMessage = {
        "role": "user",
        "content": (
            f"Generate a complete classification prompt using the following structure:\n\n"
            f"1. RoleAssignment: {RoleAssignment}\n"
            f"2. Use '---' as delimiter between each section\n"
            f"3. PerspectiveSetting: Generate suitable perspective/viewpoint guidance\n"
            f"4. ContextInfo: Generate relevant context information\n"
            f"5. TaskInstruction: Generate clear task instructions for classification\n"
            f"6. LabelSetDefinition: Generate label definitions"
            + (f" for these labels: {', '.join(ClassificationLabels)}" if ClassificationLabels else "")
            + f"\n7. OutputFormat: Generate output format instructions\n"
            f"8. ReasoningDirective: Generate reasoning/thinking instructions\n"
            f"9. FewShotBlock: Generate few-shot examples for the classification task\n\n"
            f"Output ONLY the finished prompt with '---' as delimiters between sections. "
            f"Do not include section names or numbers in the output."
        )
    }

    MaxRetries = 10
    for Attempt in range(MaxRetries):
        try:
            Response = Client.chat.completions.create(
                model = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages = [SystemMessage, UserMessage],
                temperature = 1,
            )

            PromptResult = Response.choices[0].message.content.strip()
            return ParsePromptWithDelimiter(PromptResult)
        except Exception as e:
            if Attempt == MaxRetries - 1:
                raise Exception(f"Failed to generate valid prompt after {MaxRetries} attempts. Last error: {str(e)}")
            continue


def ParsePromptWithDelimiter(PromptText: str) -> dict:
    """Parse a prompt string that uses a consistent delimiter between slots."""
    PromptLines = PromptText.splitlines()
    
    # The second line is the delimiter token
    DelimiterToken = PromptLines[1]
    
    # Split the entire text by the delimiter
    Segments = PromptText.split(DelimiterToken)
    
    # Clean up segments (remove leading/trailing whitespace)
    Segments = [Segment.strip() for Segment in Segments if Segment.strip()]
    
    ParsedPrompt = {
        "RoleAssignment": Segments[0],
        "PerspectiveSetting": Segments[1],
        "ContextInfo": Segments[2],
        "TaskInstruction": Segments[3],
        "LabelSetDefinition": Segments[4],
        "OutputFormat": Segments[5],
        "ReasoningDirective": Segments[6],
        "FewShotBlock": Segments[7],
    }

    return ParsedPrompt

def CombineString(DictPrompt: Dict) -> str:
    """Combine all string values from a prompt dictionary into a single string."""
    CombinedString = ""
    for Value in DictPrompt.values():
        if isinstance(Value, str):
            CombinedString += Value + "\n"
    return CombinedString