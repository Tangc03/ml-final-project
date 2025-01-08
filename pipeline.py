import numpy as np
from utils.LLL import LLL_reduction
from utils.new_search_fixed import search_nearest_point, sign
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
    这里直接调用给出的 LLL_reduction。
    注意：LLL_reduction 输入输出都是“列表的列表”形式，这里需要做适配。
    """
    basis_list = B.tolist()
    reduced_list = LLL_reduction(basis_list)
    return np.array(reduced_list)

def ORTH(B):
    """
    Orthogonal transformation:
    用正交变换(旋转/反射等)将 B 转化为一个下三角、且对角元素为正的方阵。
    使用 QR 分解 + 调整正负号的做法。
    """
    Q, R = np.linalg.qr(B.T)
    B_new = R.T.copy()
    n = B_new.shape[0]
    for i in range(n):
        if B_new[i, i] < 0:
            B_new[i, :] = -B_new[i, :]
    return B_new

def CLP(B, x):
    """
    Closest Lattice Point function:
    给定生成矩阵 B 和目标点 x，返回最近晶格点所对应的整数向量 u_hat ∈ Z^n。
    使用 new_search_fixed.py 的 search_nearest_point 方法。
    """
    Q, R = np.linalg.qr(B.T)
    L = R.T.copy()
    r = Q.T @ x

    u_hat_full = search_nearest_point(L.shape[0], L, r)  # n, L, r
    u_hat = np.round(u_hat_full[1:]).astype(int)

    return u_hat  # 返回 Z^n 中的整向量

def save_B_matrix_plot(B, iteration, output_dir='B_plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(6, 5))
    plt.imshow(B, cmap='viridis', aspect='auto')
    plt.colorbar(label='值')
    plt.title(f'B 矩阵 at Iteration {iteration}')
    plt.xlabel('列索引')
    plt.ylabel('行索引')
    plt.savefig(os.path.join(output_dir, f'B_iter_{iteration}.png'))
    plt.close()

def plot_B_matrix(B):
    plt.figure(figsize=(8, 6))
    plt.imshow(B, cmap='viridis', aspect='auto')
    plt.colorbar(label='值')
    plt.title('最终得到的 B 矩阵')
    plt.xlabel('列索引')
    plt.ylabel('行索引')
    plt.show()

def plot_norms_history(norms_history):
    norms_history = np.array(norms_history)
    plt.figure(figsize=(10, 6))
    for i in range(norms_history.shape[1]):
        plt.plot(norms_history[:, i], label=f'行 {i+1} 的范数')
    plt.xlabel('RED+ORTH 迭代次数')
    plt.ylabel('行向量范数')
    plt.title('B 矩阵行向量范数的变化')
    plt.legend()
    plt.show()

# ============ 主算法：迭代构造晶格基 ============

def iterative_lattice_construction(n,
                                   T=10,
                                   Tr=3,
                                   mu0=1.0,
                                   nu=2.0,
                                   tqdm_update_freq=1000,
                                   save_plots=False,
                                   output_dir='B_plots'):
    """
    按题目伪代码，实现“迭代构造晶格基”
    参数:
      n: 维度
      T: 主循环次数 (for t in [0..T-1])
      Tr: 每多少次迭代进行一次重新约减 + 正交化
      mu0, nu: 用于更新步长 μ 的参数 (μ = mu0 * nu^(-t/(T-1)))
      tqdm_update_freq: 每多少次迭代更新一次进度条
      save_plots: 是否保存 B 矩阵的图像
      output_dir: 图像保存目录
    返回:
      B: 迭代后得到的生成矩阵
      norms_history: 记录每次 RED 后 B 矩阵的行范数
    """

    norms_history = []

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

    # 记录初始的范数
    norms = np.linalg.norm(B, axis=1)
    norms_history.append(norms.copy())

    # 3. 主循环
    with tqdm(total=T, desc="迭代进度") as pbar:
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

                # 记录当前 B 矩阵的行范数
                norms = np.linalg.norm(B, axis=1)
                norms_history.append(norms.copy())

                # 保存 B 矩阵的图像
                if save_plots:
                    iteration = t + 1
                    save_B_matrix_plot(B, iteration, output_dir)

            # 每 tqdm_update_freq 步更新一次进度条
            if (t + 1) % tqdm_update_freq == 0:
                pbar.update(tqdm_update_freq)

        # 更新剩余的步数
        remaining = T % tqdm_update_freq
        if remaining > 0:
            pbar.update(remaining)

    return B, norms_history

# ============ 测试/示例 ============

if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子便于演示
    n = 10             # 维度
    T = int(1e5)       # 迭代次数，示例中使用较小的值
    Tr = 100           # 每隔多少步做一次 RED+ORTH
    mu0 = 0.01
    nu = 500.0

    # 启用保存图像
    B_final, norms_history = iterative_lattice_construction(
        n, T, Tr, mu0, nu, save_plots=True, output_dir='B_plots'
    )
    print("最终得到的 B 矩阵：")
    print(B_final)
    plot_B_matrix(B_final)
    plot_norms_history(norms_history)
