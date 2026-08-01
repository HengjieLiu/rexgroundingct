# ASD-040 artifact ownership

The manifest separates the training-safe censorship ledger from the
evaluator-only hidden-mask ledger. Hidden masks remain external runtime
evidence and are never copied into workspace results. Large features,
checkpoints, logits, and predictions stay outside Git.
