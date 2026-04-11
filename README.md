
# Adversarial Example Generation Model via Variable Renaming

## Introduction
This repository contains the core component of a robustness evaluation benchmark for code vulnerability detection models. By automatically generating adversarial examples through semantics-preserving variable renaming, this tool tests and evaluates the resilience of machine learning models (such as CodeBERT) against structural and semantic perturbations in source code.

## Key Features
Based on the configuration architecture, this framework supports a multi-layered adversarial attack and defense pipeline:
* **Dual Candidate Generation:**
    * **Lightweight Generator:** Utilizes FastText and FAISS indexes for $O(1)$ fast semantic neighbor retrieval.
    * **Heavyweight Generator:** Leverages Masked Language Modeling (MLM) via `microsoft/codebert-base-mlm` for highly context-aware and type-aware candidate predictions.
* **Advanced Optimizers:** Supports both **Greedy Search** and **Genetic Algorithms (GA)** to find the optimal adversarial substitution combination.
* **RNNS Ranking:** Prioritizes vulnerable target variables to narrow down the search space during Genetic Algorithm execution.
* **Robustness Evaluation (Smoother):** Implements a randomized smoothing mechanism with Monte Carlo sampling and majority voting to evaluate model defenses against adversarial inputs.
* **Target Language:** Fully parses and analyzes **C/C++** source code via AST.

## Getting Started

### 1. Model Preparation
Before running the attacks, you must fine-tune the base vulnerability detection model. 
Run the initialization script to train the model on your dataset:
```bash
python init_env.py
```

### 2. Testing Base Accuracy
To verify the performance of your fine-tuned model before attacking it, run the testing script:
```bash
python test.py
```
*Note: The current baseline accuracy is approximately 60%. If you wish to achieve a higher accuracy, it is highly recommended to provide a larger dataset and increase the number of training epochs inside `init_env.py`.*

### 3. Running the Attack
Once the model is fine-tuned and verified, execute the main adversarial generation pipeline:
```bash
python main.py --config config.yaml --mode binary
```
**`mode`**: Decide which way should the model run, `multi` or `binary`.

## Configuration Guide (`config.yaml`)
The pipeline is highly customizable via the `config.yaml` file. Key sections include:

* **`run_params`**: Control the attack mode, dataset path (`big_vul.parquet`), number of samples, and the core algorithm (`greedy` or `genetic`). Toggle `use_majority_voting` to enable/disable the defense simulation.
* **`heavyweight_candidate` / `lightweight_candidate`**: Adjust the thresholds for semantic similarity, context ratios, and the top-$K$ limits for masked language model predictions and FAISS retrievals.
* **`genetic_algorithm`**: Tune hyperparameters for the GA optimizer, including population size, mutation rates, and stagnation limits.
* **`smoother`**: Configure the defense mechanism by setting the number of smoothing samples, variance thresholds, and replacement probabilities.

## Output
By default, all original datasets, generated adversarial test sets, and attack success rate (ASR) matrices are saved to the `./results` directory as configured in the `global` parameters.