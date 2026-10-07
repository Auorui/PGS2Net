"""Model factories used in the RPF ablation studies.

All variants preserve the PGS2Net-S width and depth configuration. They differ
only in RPF placement, parameter learnability, or radial shape function, making
the resulting comparisons controlled and directly interpretable.

命名说明：``ep1``-``ep8`` 对应论文中的 stage-wise placement settings。
"""

from .pgs2net import PGS2Net


# -----------------------------------------------------------------------------
# Stage-wise RPF placement ablation
# Each Boolean corresponds to encoder stages E1, E2, and E3, respectively.
# -----------------------------------------------------------------------------
def PGS2Net_ep1():
    """Enable RPF only in encoder stage E1."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  False, False])

def PGS2Net_ep2():
    """Enable RPF only in encoder stage E2."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, True,  False])

def PGS2Net_ep3():
    """Enable RPF only in encoder stage E3."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, False, True])

def PGS2Net_ep4():
    """Enable RPF in encoder stages E1 and E2."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  True,  False])

def PGS2Net_ep5():
    """Enable RPF in encoder stages E2 and E3."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, True,  True])

def PGS2Net_ep6():
    """Enable RPF in encoder stages E1 and E3."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  False, True])

def PGS2Net_ep7():
    """Enable RPF in all three encoder stages (final paper setting)."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  True,  True])

def PGS2Net_ep8():
    """Disable RPF in all stages (no-RPF baseline)."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, False, False])


# -----------------------------------------------------------------------------
# RPF parameter learnability ablation
# -----------------------------------------------------------------------------
def PGS2Net_ep7_P1():
    """P1: freeze alpha, k, and r0 to use a fully static fitted prior."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=False, learn_k=False, learn_r0=False)

def PGS2Net_ep7_P2():
    """P2: learn alpha while keeping the fitted shape parameters fixed."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=True, learn_k=False, learn_r0=False)

def PGS2Net_ep7_P3():
    """P3: learn alpha, k, and r0 jointly (default ep7 configuration)."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=True, learn_k=True, learn_r0=True)

def PGS2Net_ep7_P4():
    """P4: learn the radial shape while freezing modulation strength alpha."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=False, learn_k=True, learn_r0=True)


# -----------------------------------------------------------------------------
# Radial shape-function ablation
# -----------------------------------------------------------------------------
def PGS2Net_ep7_A1():
    """A1: bounded antisymmetric tanh profile (final paper setting)."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='tanh')

def PGS2Net_ep7_A2():
    """A2: unbounded linear profile used to test the effect of saturation."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='linear')

def PGS2Net_ep7_A3():
    """A3: bounded sigmoid-derived profile with a softer transition."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='sigmoid')

def PGS2Net_ep7_A5():
    """A5: single-sided piecewise profile for high-frequency enhancement only."""
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='piecewise')
