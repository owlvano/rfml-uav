# classifier-logits-output Specification

## Purpose

Fixes the output contract of the classifier networks: training and scoring code receive raw logits, while deployed ONNX artifacts keep returning probabilities.

## Requirements

### Requirement: Models return logits
Every classifier built through the model factory SHALL return unnormalised logits of shape `(batch, n_classes)` from its forward pass, with no softmax applied.

#### Scenario: Output is not a distribution
- **WHEN** a CNN or RNN forward pass runs on a batch
- **THEN** output rows are not constrained to sum to 1, and applying softmax to them yields the class probabilities

#### Scenario: Energy score computable
- **WHEN** a consumer computes `-logsumexp(output, dim=1)`
- **THEN** it obtains the energy score directly, without inverting a softmax

### Requirement: Old checkpoints remain loadable
State dicts saved before this change SHALL load into the updated models without key or shape errors, and their argmax predictions SHALL be unchanged.

#### Scenario: Legacy checkpoint
- **WHEN** `checkpoints/v2/checkpoint_wpd_best.pt` is loaded into the updated CNN
- **THEN** loading succeeds and argmax predictions equal those of the pre-change model on the same inputs

### Requirement: ONNX export emits probabilities
The ONNX export SHALL apply softmax inside the exported graph and SHALL keep input name `input` and output name `probs` with a dynamic batch axis.

#### Scenario: Mobile contract preserved
- **WHEN** an exported ONNX model runs on a batch
- **THEN** output `probs` has rows summing to 1 (within 1e-5), with the same names and shapes as before this change

### Requirement: Optional plain linear logit tail
The model factory SHALL accept an option that builds the CNN with its final linear layer as the output (no normalisation or activation on the logits). Without the option, the CNN SHALL be built exactly as before.

#### Scenario: Default unchanged
- **WHEN** a CNN is built through the factory without the option
- **THEN** its state-dict keys equal those of the pre-change CNN, and `checkpoints/v2/checkpoint_wpd_best.pt` and `checkpoints/paper3/source_wpd_cnn.pt` load without errors

#### Scenario: Linear tail
- **WHEN** a CNN is built with the option
- **THEN** its output equals the final linear layer's output, and the factory still returns logits of shape `(batch, n_classes)`
