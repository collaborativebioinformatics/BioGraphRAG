"""Seeding utility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int | None = None) -> int:
    if seed is None:
    env_seed = os.environ.get("DUAL_KG_RAG_SEED")
        seed = int(env_seed) if env_seed else 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
"""Seeding utility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int | None = None) -> int:
    if seed is None:
        env_seed = os.environ.get("BIO_KG_SEED")
        seed = int(env_seed) if env_seed else 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed




