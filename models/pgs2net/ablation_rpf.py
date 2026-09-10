from .pgs2net import PGS2Net

#  Layout ablation (Placement, ep1-ep8)
def PGS2Net_ep1():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  False, False])

def PGS2Net_ep2():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, True,  False])

def PGS2Net_ep3():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, False, True])

def PGS2Net_ep4():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  True,  False])

def PGS2Net_ep5():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, True,  True])

def PGS2Net_ep6():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  False, True])

def PGS2Net_ep7():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True,  True,  True])

def PGS2Net_ep8():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[False, False, False])

# Ablation of Parameter Learnability
# P1: Fully frozen (pure static prior)
def PGS2Net_ep7_P1():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=False, learn_k=False, learn_r0=False)

# P2: Only train alpha, freeze k/r0 (recommended setting: shape derived from physics, only amplitude adaptive)
def PGS2Net_ep7_P2():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=True, learn_k=False, learn_r0=False)

# P3: All Can Be Learned (= Default ep2)
def PGS2Net_ep7_P3():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=True, learn_k=True, learn_r0=True)

# P4: Only learn k/r0, freeze alpha (Control: Only learn shape, not strength)
def PGS2Net_ep7_P4():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=False, learn_k=True, learn_r0=True)

# Activation Function Ablation
# A1: tanh (= Default ep2, antisymmetric bounded)
def PGS2Net_ep7_A1():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='tanh')

# A2: linear (Unsaturation, Boundedness of Contrast)
def PGS2Net_ep7_A2():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='linear')

# A3: sigmoid (Antisymmetric, Transition Softer)
def PGS2Net_ep7_A3():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='sigmoid')

# A5: piecewise (High-frequency Enhancement Only)
def PGS2Net_ep7_A5():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='piecewise')













