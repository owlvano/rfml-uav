# Spec Delta

## Purpose

Standard label-free adaptation methods (AdaBN, TENT, naive pseudo-labelling) as reusable operations on a trained BatchNorm classifier. Paper-3 notebooks use them as the baselines that OG-TCST is compared against.

## ADDED Requirements

### Requirement: Label-free input
Every adaptation method SHALL take a trained model and unlabeled target features only. No method SHALL accept or read target labels.

#### Scenario: Signature
- **WHEN** a method is called
- **THEN** its inputs are the model, the target feature matrix and method hyperparameters, with no label argument

### Requirement: Source model untouched
Every method SHALL return an adapted copy and leave the given model's parameters and buffers unchanged.

#### Scenario: Source weights after adaptation
- **WHEN** any method has run
- **THEN** the source model's state is bit-identical to its state before the call

### Requirement: AdaBN changes only normalisation statistics
AdaBN SHALL replace the running mean and variance of every BatchNorm layer with statistics estimated over the whole adapt set, layer by layer, and SHALL change no trainable parameter.

#### Scenario: Parameters unchanged
- **WHEN** AdaBN is applied
- **THEN** every trainable parameter equals the source value, and the BatchNorm running statistics differ

#### Scenario: Statistics are the adapt-set statistics
- **WHEN** AdaBN is applied to a model whose first layer is BatchNorm
- **THEN** that layer's running mean equals the adapt-set mean of its input within 1e-4

### Requirement: TENT updates only BatchNorm affine parameters
TENT SHALL minimise the mean prediction entropy on the adapt set. Its gradient updates SHALL touch only the BatchNorm weight and bias. After the updates, its running statistics SHALL be re-estimated on the adapt set, as in AdaBN.

#### Scenario: Update scope
- **WHEN** TENT is applied
- **THEN** every parameter outside the BatchNorm affine parameters equals its source value

#### Scenario: Entropy decreases
- **WHEN** TENT runs with a positive learning rate on a shifted synthetic set
- **THEN** the mean prediction entropy on the adapt set is lower after adaptation than before

### Requirement: Pseudo-labelling on confident source predictions
Pseudo-labelling SHALL label adapt windows with the source model's argmax once, before any update. It SHALL keep only windows whose max-softmax is ≥ τ and fine-tune all parameters with cross-entropy on them. If fewer than one batch of windows passes τ, it SHALL return the unchanged source copy and say so.

#### Scenario: Labels fixed before training
- **WHEN** pseudo-labelling runs for several epochs
- **THEN** the pseudo-labels used in every epoch are those of the source model, not of the model being trained

#### Scenario: Too few confident windows
- **WHEN** fewer windows than one batch pass τ
- **THEN** the returned model equals the source and the result reports zero selected windows

#### Scenario: Selection reported
- **WHEN** pseudo-labelling runs
- **THEN** the result reports how many windows were selected per pseudo-class

### Requirement: Deterministic adaptation
Given the same model, adapt set, hyperparameters and seed, a method SHALL produce the same adapted predictions.

#### Scenario: Repeat run
- **WHEN** a method is run twice with identical inputs and seed
- **THEN** the argmax predictions on a fixed test set are identical

### Requirement: Cost metadata
Every method SHALL report the number of forward passes and backward passes it performed per adapt window, and the number of parameters it changed.

#### Scenario: Cost of AdaBN
- **WHEN** AdaBN is applied
- **THEN** it reports one forward pass per window, zero backward passes and zero updated trainable parameters

#### Scenario: Cost of TENT
- **WHEN** TENT runs for E epochs
- **THEN** it reports E + 1 forward and E backward passes per window (the extra forward pass re-estimates the statistics), and the BatchNorm affine parameter count as updated parameters
