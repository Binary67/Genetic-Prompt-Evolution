# Genetic-Prompt-Evolution

This repository shows **how to use Genetic Programming (GP) to evolve Large-Language-Model prompts** for text classification tasks. Originally designed for binary classification of employee feedback, it now supports **any classification task** with configurable labels, data columns, and task descriptions.

The project contains three key ideas:

1. **Prompt Skeleton** – A flexible template whose *slots* can be mutated, swapped, or removed by GP.
2. **`GeneratePromptSample()`** – A Python helper that asks Azure OpenAI to fill the skeleton with sensible defaults so you can bootstrap your GP population.
3. **`ClassificationTaskConfig`** – A configuration class that makes the system generic for any classification task.

## 🚀 Key Features

- **Async Azure OpenAI Integration**: Massively improved performance through concurrent API calls
- **Intelligent Rate Limiting**: Exponential backoff strategy with configurable concurrent request limits
- **Generic Classification Support**: Works with any text classification task
- **Progress Tracking**: Real-time feedback during evolution process

## 1  Prompt Skeleton

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

* **Fixed parts**: the `{TaskDescription}` (configurable per task) and the `{Text}` placeholder remain untouched.
* **Mutable slots**: everything in curly braces except `{Text}` and `{TaskDescription}` can be added, removed, or edited by GP.
