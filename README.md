# Genetic-Prompt-Evolution

This repository demonstrates **how to use Genetic Programming (GP) to evolve Large-Language-Model prompts** for text classification tasks. The system automatically evolves and optimizes prompts to achieve better classification accuracy through genetic operations like crossover and mutation.

## 🚀 Key Features

- **Genetic Programming for Prompt Evolution**: Automatically evolves prompts using genetic operations (crossover, mutation, tournament selection)
- **Azure OpenAI Integration**: Uses Azure OpenAI API for prompt generation and evaluation
- **Flexible Prompt Structure**: Modular prompt skeleton with 8 customizable components
- **Multi-class Classification Support**: Works with any number of classification labels
- **Automatic Prompt Generation**: Bootstraps initial population using LLM-generated prompts
- **Elite Preservation**: Keeps best-performing prompts across generations
- **Progress Tracking**: Real-time feedback during evolution process
- **Evolution History**: Saves complete evolution history and best prompts to JSON
- **Validation Evaluation**: Tests best evolved prompt on separate validation data

## 📋 How It Works

### 1. Prompt Structure

The system uses a modular prompt skeleton with 8 customizable components:

```text
{RoleAssignment}
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
```

**Components:**
- **RoleAssignment**: Defines the AI's role/expertise
- **PerspectiveSetting**: Sets the viewpoint or approach
- **ContextInfo**: Provides relevant background information
- **TaskInstruction**: Clear classification instructions
- **LabelSetDefinition**: Defines each classification label
- **OutputFormat**: Specifies expected output format
- **ReasoningDirective**: Guides reasoning process
- **FewShotBlock**: Contains example classifications

### 2. Genetic Evolution Process

1. **Initial Population**: Generate N prompts using Azure OpenAI
2. **Evaluation**: Test each prompt on training data to calculate fitness (accuracy)
3. **Selection**: Use tournament selection to choose parents
4. **Crossover**: Combine components from two parent prompts
5. **Mutation**: Randomly modify prompt components using LLM
6. **Elite Preservation**: Keep best prompts for next generation
7. **Repeat**: Continue for specified number of generations

### 3. Core Components

- **GPPopulation.py**: Generates initial prompt population
- **GPOperation.py**: Implements genetic operations (crossover, mutation, selection)
- **GPEvaluate.py**: Evaluates prompts and calculates fitness scores
- **GPEvolution.py**: Main evolution loop and orchestration
- **PromptStructure.py**: Prompt generation and parsing utilities

### Parameters

- **InputData**: DataFrame with 'text' and 'label' columns
- **ClassificationLabels**: List of classification labels
- **PopulationSize**: Number of prompts per generation (default: 2)
- **NumGenerations**: Number of evolution generations (default: 2)
- **MutationRate**: Probability of mutation (0-1, default: 0.3)
- **EliteSize**: Number of best prompts to preserve (default: 2)
- **RoleAssignment**: Initial role description for the AI

## 📊 Output

The system provides:

1. **Real-time Progress**: Shows fitness scores for each generation
2. **Evolution History**: Saved to `evolution_history.json` with:
   - Best prompt from each generation
   - Fitness progression
   - Final best prompt structure
3. **Validation Results**: 
   - Overall accuracy
   - Per-class accuracy
   - Misclassified examples

## 🔧 Advanced Features

### Custom Mutation
The mutation process uses Azure OpenAI to intelligently modify prompt components while maintaining coherence and purpose.

### Tournament Selection
Implements tournament selection with configurable tournament size for parent selection.

### Flexible Classification
Supports any classification task - just provide your labels and data:
- Sentiment analysis
- Intent classification  
- Topic categorization
- Custom business classifications

## 📁 Project Structure

```
├── main.py                 # Example usage
├── GPEvolution.py         # Main evolution loop
├── GPPopulation.py        # Population generation
├── GPOperation.py         # Genetic operations
├── GPEvaluate.py          # Evaluation and fitness
├── PromptStructure.py     # Prompt utilities
└── README.md              # Documentation
```