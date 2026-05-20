"""LVDRS-style kick-only training task.

Reproduces the AMP + multi-critic + mirror-symmetry + virtual-perception
pipeline from the LVDRS paper (arXiv:2511.03996), scoped down to a single
kick skill (no long-range ball search).
"""

from mjlab.tasks.kick import mdp

__all__ = ["mdp"]
