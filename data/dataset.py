"""Dataset dispatcher — selects the loader based on config.DATASET.
All arms import load_sample from here, so switching datasets is one config line.
"""
import config

def load_sample(*a, **k):
    if config.DATASET == "musique":
        from data.musique_loader import load_sample as ls
    else:
        from data.loader import load_sample as ls
    return ls(*a, **k)
