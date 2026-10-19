def get_alpha_beta_state(checkpoint):
    if "alpha_beta" in checkpoint:
        return checkpoint["alpha_beta"]
    if "alpha_beat" in checkpoint:
        return checkpoint["alpha_beat"]
    raise KeyError("Checkpoint is missing the alpha_beta (or legacy alpha_beat) state")

