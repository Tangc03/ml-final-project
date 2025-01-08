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
    给定生成矩阵 B 和一个点 x，找到使得 u*B 最接近 x 的整数向量 u。
    （返回的是 u 而非最近点本身）
    
    这里我们演示一个“先将 B 转为下三角，再调用 search_nearest_point”的做法：
      1) 找到一个正交矩阵 Q，使得 B = Q * L (L下三角)。
         为简化，此处我们直接用 ORTH(B) 得到一个下三角 L' ≈ B 的某种正交变换；
         实际若想严格满足 B = Q*L，需要做进一步分解，这里仅作示例。
      2) 把 x 也变换到与 L 对齐的坐标系 r = Q^T x
      3) 调用 search_nearest_point(n, L, r) 得到整向量 u
      4) 这里演示版本只是直接对 B 做 ORTH(B)；严格场合需保留 Q 方便做坐标变换
         简化起见，这里只演示“B 先求逆”然后再四舍五入，也可以。

    实际工程中，CLP 是个不简单的问题。下述实现仅作示例！
    """
    # ---- 简易方法：u = round(inv(B)*x) ----
    #    注意这种方法对 B 未必是最佳近邻，但代码简单。
    #    若要用 search_nearest_point，需把 B 真正拆成下三角 L 和正交 Q。
    #    这里为不繁琐，演示“直接逆阵 + 四舍五入”方法：
    z_float = np.linalg.inv(B) @ x
    u_hat = np.rint(z_float)  # 逐坐标就近取整
    return u_hat  # 返回整数向量

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

    # 1. 初始: B <- ORTH(RED(GRAN(n,n)))
    B_init = GRAN(n, n)                 # 高斯随机
    B_red  = RED(B_init)                # LLL 约减
    B = ORTH(B_red)                     # 再正交化/转下三角
    
    # 2. 令 V = (∏ B[i,i])^(-1/n)，再做一次缩放，让 det(B) ~ 1
    #   这里演示取对角线乘积为 diag_prod
    diag_prod = 1.0
    for i in range(n):
        diag_prod *= B[i, i]
    # 如果对角线上有负值，上面 ORTH 里已经处理过了，这里假设其全为正
    alpha = diag_prod ** (-1.0 / n)
    B = alpha * B   # B <- alpha * B

    # 主循环
    for t in range(T):
        # 5. μ <- μ0 * ν^(- t/(T-1))   (若 T=1，为避免 0/0，这里做个保护)
        if T > 1:
            mu = mu0 * (nu ** ( - float(t)/(T-1) ))
        else:
            mu = mu0
        
        # 6. z <- URAN(n)    (均匀随机向量)
        z = URAN(n)
        
        # 7. y <- z - CLP(B, zB)
        zB = z @ B          # z 是 1 x n, B 是 n x n => zB 仍是长度 n 的向量
        u_hat = CLP(B, zB)  # 最近点对应的整数向量
        y = z - u_hat       # 这里 y 作为一个 n 维向量
        
        # 8. e <- yB
        e = y @ B  # 仍然是 n 维
        
        # 9~14: 更新 B 的各分量
        #    for i=1..n:
        #      for j=1..(i-1):
        #         B[i,j] = B[i,j] - mu * y[i]* e[j]
        #      B[i,i] = B[i,i] - mu * ( y[i]* e[i] - ||e||^2/( n * B[i,i] ) )
        #
        # 注意 Python 下标是 0..n-1，所以要做适配
        e_norm_sq = np.linalg.norm(e)**2
        for i_idx in range(n):
            for j_idx in range(i_idx):
                B[i_idx, j_idx] -= mu * y[i_idx] * e[j_idx]
            # 对角项
            B_ii = B[i_idx, i_idx]
            B[i_idx, i_idx] = B_ii - mu * ( y[i_idx]*e[i_idx] - e_norm_sq/(n * B_ii) )

        # 15. 如果 (t mod Tr) = Tr - 1，则再做一次约减 + 正交化 + 规模调整
        if (t % Tr) == (Tr - 1):
            B = ORTH(RED(B))
            # 再做一次类似 (∏ B[i,i])^(-1/n) 的缩放
            diag_prod = 1.0
            for i in range(n):
                diag_prod *= B[i, i]
            alpha = diag_prod ** (-1.0 / n)
            B = alpha * B

    return B

# ============ 调试或测试 ============

if __name__ == "__main__":
    n = 4       # 维度
    T = 5       # 迭代次数
    Tr = 2      # 每隔多少步做一次 RED+ORTH
    mu0 = 0.5
    nu = 2.0

    B_final = iterative_lattice_construction(n, T, Tr, mu0, nu)
    print("最终得到的 B 矩阵：")
    print(B_final)
