import pandas as pd
from PromptStructure import GeneratePromptSample, CombineString
from GPEvaluate import EvaluatePrompt, CalculateFitnessScore
from GPPopulation import GeneratePopulation
from GPOperation import Crossover, Mutation, TournamentSelection
import random

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
    # Genetic Algorithm Parameters
    ClassificationLabels = ["compliment", "development"]
    PopulationSize = 3
    NumGenerations = 3
    MutationRate = 0.3
    EliteSize = 2  # Number of best individuals to carry forward
    
    print(f"Starting Genetic Prompt Evolution")
    print(f"Population Size: {PopulationSize}")
    print(f"Generations: {NumGenerations}")
    print(f"Mutation Rate: {MutationRate}")
    print(f"Elite Size: {EliteSize}\n")
    
    # Generate initial population
    print(f"Generating initial population of {PopulationSize} prompts")
    CurrentPopulation = GeneratePopulation(
        PopulationSize=PopulationSize,
        RoleAssignment="You are an expert HR analyst specializing in employee feedback classification.",
        ClassificationLabels=ClassificationLabels
    )
    
    # Track best fitness across generations
    GenerationBestFitness = []
    AllTimeBestFitness = 0
    AllTimeBestPrompt = None
    
    # Evolution loop
    for Generation in range(NumGenerations):
        print(f"\n{'='*50}")
        print(f"GENERATION {Generation + 1}")
        print(f"{'='*50}")
        
        # Evaluate current population
        FitnessScores = []
        for i, PromptDict in enumerate(CurrentPopulation):
            print(f"\nEvaluating Individual {i+1}")
            ResultDF = EvaluatePrompt(ValidationData, PromptDict)
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
        print(f"{Value[:100]}" if len(Value) > 100 else Value)