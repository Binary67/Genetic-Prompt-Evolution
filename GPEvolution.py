import pandas as pd
from PromptStructure import GeneratePromptSample, CombineString
from GPEvaluate import EvaluatePrompt, CalculateFitnessScore
from GPPopulation import GeneratePopulation
from GPOperation import Crossover, Mutation, TournamentSelection
import random
import json

def RunEvolution(
    InputData: pd.DataFrame,
    ClassificationLabels: list,
    PopulationSize: int = 2,
    NumGenerations: int = 2,
    MutationRate: float = 0.3,
    EliteSize: int = 2,
    RoleAssignment: str = "You are an expert HR analyst specializing in employee feedback classification."
):
    """
    Run genetic algorithm to evolve prompts for text classification
    
    Args:
        TrainingData: DataFrame with 'text' and 'label' columns for training
        InputData: DataFrame with 'text' and 'label' columns for validation
        ClassificationLabels: List of classification labels
        PopulationSize: Number of prompts in each generation
        NumGenerations: Number of generations to evolve
        MutationRate: Probability of mutation (0-1)
        EliteSize: Number of best individuals to carry forward
        RoleAssignment: Role description for the LLM
    
    Returns:
        dict: Best prompt found during evolution
    """
    print(f"Starting Genetic Prompt Evolution")
    print(f"Population Size: {PopulationSize}")
    print(f"Generations: {NumGenerations}")
    print(f"Mutation Rate: {MutationRate}")
    print(f"Elite Size: {EliteSize}\n")
    
    # Generate initial population
    print(f"Generating initial population of {PopulationSize} prompts")
    CurrentPopulation = GeneratePopulation(
        PopulationSize=PopulationSize,
        RoleAssignment=RoleAssignment,
        ClassificationLabels=ClassificationLabels
    )
    
    # Track best fitness across generations
    GenerationBestFitness = []
    AllTimeBestFitness = 0
    AllTimeBestPrompt = None
    
    # Track best individuals from each generation
    GenerationHistory = []
    
    # Evolution loop
    for Generation in range(NumGenerations):
        print(f"\n{'='*50}")
        print(f"GENERATION {Generation + 1}")
        print(f"{'='*50}")
        
        # Evaluate current population
        FitnessScores = []
        for i, PromptDict in enumerate(CurrentPopulation):
            print(f"\nEvaluating Individual {i+1}")
            ResultDF = EvaluatePrompt(InputData, PromptDict, ClassificationLabels)
            Fitness = CalculateFitnessScore(ResultDF)
            FitnessScores.append(Fitness)
            print(f"Fitness: {Fitness:.2f}%")
            
            # Track all-time best
            if Fitness > AllTimeBestFitness:
                AllTimeBestFitness = Fitness
                AllTimeBestPrompt = PromptDict.copy()
        
        # Sort population by fitness
        PopulationWithFitness = list(zip(CurrentPopulation, FitnessScores))
        PopulationWithFitness.sort(key=lambda x: x[1], reverse=True)
        
        # Report generation statistics
        GenerationBest = PopulationWithFitness[0][1]
        GenerationAverage = sum(FitnessScores) / len(FitnessScores)
        GenerationBestFitness.append(GenerationBest)
        
        # Save best individual from this generation
        BestPromptDict = PopulationWithFitness[0][0]
        BestIndividual = {
            "Generation": Generation + 1,
            "Fitness": GenerationBest,
            "PromptDict": BestPromptDict,
            "PromptText": CombineString(BestPromptDict)
        }
        GenerationHistory.append(BestIndividual)
        
        print(f"\nGeneration {Generation + 1} Summary:")
        print(f"Best Fitness: {GenerationBest:.2f}%")
        print(f"Average Fitness: {GenerationAverage:.2f}%")
        print(f"All-Time Best: {AllTimeBestFitness:.2f}%")
        
        # Skip evolution on last generation
        if Generation == NumGenerations - 1:
            break
        
        # Create next generation
        NextPopulation = []
        
        # Elitism: Keep best individuals
        for i in range(EliteSize):
            NextPopulation.append(PopulationWithFitness[i][0].copy())
        
        # Generate offspring to fill rest of population
        while len(NextPopulation) < PopulationSize:
            # Tournament selection for parents
            Parent1 = TournamentSelection(CurrentPopulation, FitnessScores, TournamentSize=3)
            Parent2 = TournamentSelection(CurrentPopulation, FitnessScores, TournamentSize=3)
            
            # Crossover
            Offspring = Crossover(Parent1, Parent2)
            
            # Mutation
            Offspring = Mutation(Offspring, MutationRate, ClassificationLabels)
            
            NextPopulation.append(Offspring)
        
        CurrentPopulation = NextPopulation
    
    # Final results
    print(f"\n{'='*50}")
    print(f"EVOLUTION COMPLETE")
    print(f"{'='*50}")
    print(f"\nFitness progression across generations:")
    for i, Fitness in enumerate(GenerationBestFitness):
        print(f"Generation {i+1}: {Fitness:.2f}%")
    
    print(f"\nAll-Time Best Fitness: {AllTimeBestFitness:.2f}%")
    print(f"\nBest Prompt Structure:")
    for Key, Value in AllTimeBestPrompt.items():
        print(f"\n{Key}:")
        print(f"{Value}" if len(Value) > 100 else Value)
    
    # Save evolution history to JSON file
    EvolutionData = {
        "EvolutionSummary": {
            "PopulationSize": PopulationSize,
            "NumGenerations": NumGenerations,
            "MutationRate": MutationRate,
            "EliteSize": EliteSize,
            "AllTimeBestFitness": AllTimeBestFitness
        },
        "GenerationHistory": GenerationHistory,
        "AllTimeBestPrompt": {
            "Fitness": AllTimeBestFitness,
            "PromptDict": AllTimeBestPrompt,
            "PromptText": CombineString(AllTimeBestPrompt)
        }
    }
    
    with open("evolution_history.json", "w") as JsonFile:
        json.dump(EvolutionData, JsonFile, indent=2)
    
    print(f"\nEvolution history saved to evolution_history.json")
    
    return AllTimeBestPrompt