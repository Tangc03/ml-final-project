import numpy as np
from utils.LLL import LLL_reduction
from utils.new_search_fixed import search_nearest_point, sign

# ============ 一些基础函数 ============

def GRAN(n, m):
    """
    Gaussian randomize:
    从标准正态分布中 draw (n*m) 个点，组成一个 n x m 的矩阵
    """
    return np.random.randn(n, m)

def URAN(n):
    """
    Uniform randomize:
    从区间 [0,1) 中均匀抽取 n 个点，组成一个向量
    """
    return np.random.rand(n)

def RED(B):
    """
    Reduction:
    使用 Lenstra–Lenstra–Lovász (LLL) 算法对基 B 进行长度和近似正交的约减。
    这里直接调用题目中给出的 LLL_reduction。
    注意：LLL_reduction 输入输出都是“列表的列表”形式，这里需要做适配。
    """
    # B 可能是 numpy 数组，LLL_reduction 里直接操作列表。
    # 先转成 list of list:
    basis_list = B.tolist()
    reduced_list = LLL_reduction(basis_list)
    # 再转回 numpy array:
    return np.array(reduced_list)

def ORTH(B):
    """
    Orthogonal transformation:
    用正交变换(旋转/反射等)将 B 转化为一个下三角、且对角元素为正的方阵。
    这里为了演示，使用 QR 分解 + 调整正负号的做法：
      B^T = Q * R  =>  B = R^T * Q^T
    如果我们把 R^T 作为新 B，就可以得到一个下三角(转置后原先的 R 是上三角)。
    然后再把对角线确保为正即可。
    """
    # B^T = Q * R
    Q, R = np.linalg.qr(B.T)
    # R^T 就是下三角
    B_new = R.T.copy()
    n = B_new.shape[0]
    # 将对角线元素调成正值（如果是负的，就整行取负，从而相当于一次反射）
    for i in range(n):
        if B_new[i, i] < 0:
            B_new[i, :] = -B_new[i, :]
    return B_new

def CLP(B, x):
    """
    Closest Lattice Point function:
    给定生成矩阵 B 和目标点 x，返回最近晶格点所对应的整数向量 u_hat ∈ Z^n。
    这里必须使用 new_search_fixed.py 的 search_nearest_point 方法。
    
    步骤：
      1) 做 B^T = Q * R => B = R^T * Q^T，使得 L = R^T 是下三角。
      2) r = Q^T x 将 x 转到下三角坐标系。
      3) 调用 search_nearest_point(n, L, r) 搜索 u_hat（在其内部维度可能多 1，需截取）。
      4) 返回 u_hat (形状 n 的整数向量)。
    """
    # 1) B^T = Q * R
    Q, R = np.linalg.qr(B.T)
    # 2) L = R^T (下三角)，将 x 转到新坐标
    L = R.T.copy()
    r = Q.T @ x

    # 3) 调 search_nearest_point, 它返回一个长度 n+1 的数组 u_hat，
    #    其中 u_hat[1..n] 才是真正的整数向量 (见 new_search_fixed.py 注释)。
    u_hat_full = search_nearest_point(L.shape[0], L, r)  # n, L, r
    # 取后 n 个分量为真正的解
    # 因为 new_search_fixed.py 中定义的 u_hat 是 1-based: u_hat[1..n]
    u_hat = np.round(u_hat_full[1:]).astype(int)

    return u_hat  # 返回 Z^n 中的整向量

# ============ 主算法：迭代构造晶格基 ============

def iterative_lattice_construction(n,
                                   T=10,
                                   Tr=3,
                                   mu0=1.0,
                                   nu=2.0):
    """
    按题目伪代码，实现“迭代构造晶格基”
    参数:
      n: 维度
      T: 主循环次数 (for t in [0..T-1])
      Tr: 每多少次迭代进行一次重新约减 + 正交化
      mu0, nu: 用于更新步长 μ 的参数 (μ = μ0 * ν^(-t/(T-1)))
    返回:
      B: 迭代后得到的生成矩阵
    """

    # 1. B <- ORTH( RED( GRAN(n,n) ) )
    B_init = GRAN(n, n)   # 高斯随机初始化
    B_red  = RED(B_init)  # LLL 约减
    B = ORTH(B_red)       # 正交化为下三角

    # 2. B <- V^(-1/n) * B, 其中 V = ∏ B[i,i]
    diag_prod = 1.0
    for i in range(n):
        diag_prod *= B[i, i]
    alpha = diag_prod ** (-1.0 / n)
    B = alpha * B

    # 3. 主循环
    for t in range(T):
        # 5. 更新 μ = μ0 * ν^(-t/(T-1))
        if T > 1:
            mu = mu0 * (nu ** ( - float(t) / (T - 1) ))
        else:
            mu = mu0
        
        # 6. z <- URAN(n)
        z = URAN(n)

        # 7. u_hat <- CLP(B, zB)，然后 y = z - u_hat
        zB = z @ B           # z 是 1×n, B 是 n×n => zB 仍是 n 维向量
        u_hat = CLP(B, zB)   # 最近点对应的整向量
        y = z - u_hat        # 仍是 n 维向量

        # 8. e <- yB
        e = y @ B

        # 9~14: 更新 B[i,j]
        e_norm_sq = np.linalg.norm(e)**2
        for i_idx in range(n):
            for j_idx in range(i_idx):
                B[i_idx, j_idx] -= mu * y[i_idx] * e[j_idx]
            # 对角元素
            bii = B[i_idx, i_idx]
            B[i_idx, i_idx] = bii - mu * ( y[i_idx]*e[i_idx] - e_norm_sq / (n * bii) )

        # 15. 每隔 Tr 步做一次 B <- ORTH( RED( B ) )
        if (t % Tr) == (Tr - 1):
            B = RED(B)
            B = ORTH(B)
            # 再做一次 (∏ B[i,i])^(-1/n) 的缩放
            diag_prod = 1.0
            for i in range(n):
                diag_prod *= B[i, i]
            alpha = diag_prod ** (-1.0 / n)
            B = alpha * B

    return B

# ============ 测试/示例 ============

if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子便于演示
    n = 4      # 维度
    T = 5      # 迭代次数
    Tr = 2     # 每隔多少步做一次 RED+ORTH
    mu0 = 0.5
    nu = 2.0

    B_final = iterative_lattice_construction(n, T, Tr, mu0, nu)
    print("最终得到的 B 矩阵：")
    print(B_final)
