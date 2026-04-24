import numpy as np


# ==========================================
# 1. 基础映射函数 (Mapping Functions)
# ==========================================
def phi(x):
    """
    映射函数，使用 Sigmoid 实现。
    满足 phi(x) + phi(-x) = 1 的性质。
    """
    # 限制输入范围防止指数爆炸
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def phi_inv(y):
    """
    反映射函数，使用 Logit 实现。
    将 (0, 1) 的输入映射回实数域。
    """
    # 限制输入范围防止 log(0)
    y = np.clip(y, 1e-15, 1.0 - 1e-15)
    return np.log(y / (1.0 - y))


# ==========================================
# 2. OPNs 代数运算 (Algebraic Operations)
# ==========================================
def opns_add(alpha, beta):
    """
    根据公式 (1) 计算两个 OPNs 的加法
    alpha, beta: 形如 (u, v) 的元组或 numpy 数组
    """
    u_alpha, v_alpha = alpha
    u_beta, v_beta = beta

    res_u = phi(phi_inv(u_alpha) + phi_inv(u_beta))
    res_v = phi(phi_inv(v_alpha) + phi_inv(v_beta))
    return np.array([res_u, res_v])


def opns_multiply_scalar_minus_one(alpha):
    """
    修正版：计算 (-1) * alpha。
    根据论文性质 phi(x) + phi(-x) = 1，相反数在 OPNs 空间中的映射等价于用 1 去减。
    """
    u_alpha, v_alpha = alpha
    # 取补集，这才是真正的映射域相反数
    return np.array([1.0 - u_alpha, 1.0 - v_alpha])


def opns_subtract(alpha, beta):
    """
    根据公式 (2) 计算减法：alpha - beta = alpha + (-1)*beta
    """
    neg_beta = opns_multiply_scalar_minus_one(beta)
    return opns_add(alpha, neg_beta)


# ==========================================
# 3. 核心全序比较算子 (Total Order Relation)
# ==========================================
def opns_less_equal(alpha, beta):
    """
    根据公式 (4) 及其符号系统判断 alpha <= beta [cite: 72]
    """
    gamma = opns_subtract(alpha, beta)
    u_gamma, v_gamma = gamma[0], gamma[1]

    sum_uv = u_gamma + v_gamma

    # 增加微小的 eps 防止 numpy 浮点数计算时的精度丢失
    if sum_uv > 1.0 + 1e-9:
        # gamma < 0，即 alpha - beta < 0 => alpha <= beta
        return True
    elif np.isclose(sum_uv, 1.0, atol=1e-9):
        # 中性状态以及严格等于 0 的情况
        if u_gamma <= v_gamma + 1e-9:  # 修复点：加上等号，允许 alpha == beta
            return True
        else:
            return False
    else:
        # sum_uv < 1.0, gamma > 0
        return False


def opns_less_equal_vectorized(X_feature, theta):
    """
    向量化版本的 OPNs 全序比较算子。
    彻底消灭 Python for 循环，利用 numpy 的底层 C 语言数组广播机制实现极限加速。

    参数:
    X_feature: np.ndarray, 形状为 (N, 2)，N 个样本的特征对
    theta: np.ndarray, 形状为 (2,)，当前探测的阈值

    返回:
    np.ndarray (bool), 形状为 (N,)，如果是 True 表示该样本 <= theta
    """
    # 提取数组列 (N, )
    u_alpha, v_alpha = X_feature[:, 0], X_feature[:, 1]

    # 获取 theta 阈值的相反数 (取补集)
    u_beta, v_beta = theta[0], theta[1]
    neg_u_beta, neg_v_beta = 1.0 - u_beta, 1.0 - v_beta

    # 向量化计算 alpha + (-beta)
    # 利用我们之前写好的 phi 和 phi_inv (它们天然支持 numpy 数组运算)
    res_u = phi(phi_inv(u_alpha) + phi_inv(neg_u_beta))
    res_v = phi(phi_inv(v_alpha) + phi_inv(neg_v_beta))

    sum_uv = res_u + res_v

    # 向量化判断符号系统
    mask_negative = sum_uv > 1.0 + 1e-9
    mask_neutral_and_less = np.isclose(sum_uv, 1.0, atol=1e-9) & (res_u <= res_v + 1e-9)

    # 只要满足其一，即视为 <= 阈值
    return mask_negative | mask_neutral_and_less

#
# if __name__ == "__main__":
#     print("====== OPNs 核心数学引擎测试 ======")
#
#     # 测试 1：验证 Sigmoid 和 Logit 是否互为逆运算
#     test_val = 0.75
#     recovered_val = phi(phi_inv(test_val))
#     print(f"[测试 1] 映射可逆性: 输入 {test_val}, 还原 {recovered_val:.4f}")
#     assert np.isclose(test_val, recovered_val), "映射函数有误！"
#
#     # 构造两个符合要求的 OPNs (值必须在 0 到 1 之间)
#     # alpha 看起来偏小，beta 看起来偏大
#     alpha = np.array([0.3, 0.7])
#     beta = np.array([0.8, 0.4])
#
#     print(f"\n设定 OPNs:")
#     print(f"alpha = {alpha}")
#     print(f"beta  = {beta}")
#
#     # 测试 2：验证代数运算 (加法和减法)
#     gamma_add = opns_add(alpha, beta)
#     gamma_sub = opns_subtract(alpha, beta)
#     print(f"\n[测试 2] 代数运算:")
#     print(f"alpha + beta = [{gamma_add[0]:.4f}, {gamma_add[1]:.4f}]")
#     print(f"alpha - beta = [{gamma_sub[0]:.4f}, {gamma_sub[1]:.4f}]")
#
#     # 测试 3：验证核心全序关系 (比较大小)
#     res_alpha_le_beta = opns_less_equal(alpha, beta)
#     res_beta_le_alpha = opns_less_equal(beta, alpha)
#
#     print(f"\n[测试 3] 全序比较:")
#     print(f"判断 alpha <= beta : {res_alpha_le_beta}")
#     print(f"判断 beta <= alpha : {res_beta_le_alpha}")