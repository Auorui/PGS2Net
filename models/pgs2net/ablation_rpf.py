from .pgs2net import PGS2Net

#  布局消融 (Placement, ep1-ep8)
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

# 参数可学习性消融
# P1: 全冻结 (纯静态先验)
def PGS2Net_ep7_P1():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=False, learn_k=False, learn_r0=False)

# P2: 只学 alpha, 冻结 k/r0 (推荐设定: 形状来自物理, 仅强度自适应)
def PGS2Net_ep7_P2():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=True, learn_k=False, learn_r0=False)

# P3: 全可学 (= 默认 ep2)
def PGS2Net_ep7_P3():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=True, learn_k=True, learn_r0=True)

# P4: 只学 k/r0, 冻结 alpha (对照: 只学形状不学强度)
def PGS2Net_ep7_P4():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   learn_alpha=False, learn_k=True, learn_r0=True)

# 激活函数消融
# A1: tanh (= 默认 ep2, 反对称有界)
def PGS2Net_ep7_A1():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='tanh')

# A2: linear (无饱和, 反衬有界性)
def PGS2Net_ep7_A2():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='linear')

# A3: sigmoid (反对称, 过渡更软)
def PGS2Net_ep7_A3():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='sigmoid')

# A5: piecewise (仅高频增强)
def PGS2Net_ep7_A5():
    return PGS2Net(embed_dim=[24, 48, 96, 48, 24], depth=[2, 2, 4, 2, 2], use_rpf=[True, True, True],
                   act_type='piecewise')













