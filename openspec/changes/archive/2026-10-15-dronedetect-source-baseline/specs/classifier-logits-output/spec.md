# Spec Delta

## ADDED Requirements

### Requirement: Optional plain linear logit tail
The model factory SHALL accept an option that builds the CNN with its final linear layer as the output (no normalisation or activation on the logits). Without the option, the CNN SHALL be built exactly as before.

#### Scenario: Default unchanged
- **WHEN** a CNN is built through the factory without the option
- **THEN** its state-dict keys equal those of the pre-change CNN, and `checkpoints/v2/checkpoint_wpd_best.pt` and `checkpoints/paper3/source_wpd_cnn.pt` load without errors

#### Scenario: Linear tail
- **WHEN** a CNN is built with the option
- **THEN** its output equals the final linear layer's output, and the factory still returns logits of shape `(batch, n_classes)`
