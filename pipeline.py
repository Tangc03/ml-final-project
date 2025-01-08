import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 假设 utils.LLL.LLL_reduction 和 utils.new_search_fixed.search_nearest_point 已经实现
# 这里提供一个简单的 LLL_reduction 的占位实现
def LLL_reduction(basis_list):
    # 这里应调用实际的 LLL 约减算法
    # 作为示例，返回输入不变
    return basis_list

# 占位的 search_nearest_point 实现
def search_nearest_point(n, L, r):
    # 这里应调用实际的最近点搜索算法
    # 作为示例，返回一个全零向量
    return np.concatenate(([0], np.zeros(n)))

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

# ============ Theta 图像相关函数 ============

def count_lattice_points_exact(B, r):
    """
    确切计算 N(B, r)，适用于较低维度。
    参数:
        B (np.ndarray): 基矩阵，形状为 (n, n)
        r (float): 半径
    返回:
        int: 满足 ||uB|| <= r 的整数向量 u 的数量
    """
    n = B.shape[0]
    col_norms = np.linalg.norm(B, axis=0)
    min_col_norm = np.min(col_norms)
    max_u = int(np.ceil(r / min_col_norm))

    count = 0
    for u in product(range(-max_u, max_u + 1), repeat=n):
        u = np.array(u)
        norm = np.linalg.norm(u @ B)
        if norm <= r:
            count += 1
    return count

def count_lattice_points_monte_carlo(B, r, num_samples=100000):
    """
    使用蒙特卡洛方法估计 N(B, r)，适用于高维度。
    参数:
        B (np.ndarray): 基矩阵，形状为 (n, n)
        r (float): 半径
        num_samples (int): 采样次数
    返回:
        float: 估计的满足 ||uB|| <= r 的格点数量
    """
    n = B.shape[0]
    col_norms = np.linalg.norm(B, axis=0)
    min_col_norm = np.min(col_norms)
    k = int(np.ceil(r / min_col_norm))

    # 生成随机整数向量 u
    u_samples = np.random.randint(-k, k+1, size=(num_samples, n))
    # 计算 ||uB||
    norms = np.linalg.norm(u_samples @ B, axis=1)
    # 计算满足条件的样本比例
    count = np.sum(norms <= r)
    # 估计 N(B, r) ~ count / num_samples * (2k+1)^n
    estimated_N = (count / num_samples) * ((2 * k + 1) ** n)
    return estimated_N

def plot_theta(theta_values, r_values, method='Monte Carlo'):
    """
    绘制 Theta 图像，即 N(B, r) 与 r^2 的关系图。
    参数:
        theta_values (list of float): 记录的 N(B, r) 值
        r_values (list or np.ndarray): 对应的半径值 r
        method (str): 计算方法，'Exact' 或 'Monte Carlo'
    """
    plt.figure(figsize=(8,6))
    r_squared = np.array(r_values)**2
    plt.plot(r_squared, theta_values, marker='o', linestyle='-', label=method)
    plt.xlabel('$r^2$')
    plt.ylabel('$N(B, r)$')
    plt.title('Theta 图像')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_B_matrix(B):
    """
    绘制基矩阵 B 的热力图。
    参数:
        B (np.ndarray): 基矩阵，形状为 (n, n)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(B, cmap='viridis', aspect='auto')
    plt.colorbar(label='值')
    plt.title('最终得到的 B 矩阵')
    plt.xlabel('列索引')
    plt.ylabel('行索引')
    plt.show()
    # save the plot
    plt.savefig('B_matrix.png')

def save_B_matrix_plot(B, iteration, output_dir='B_plots'):
    """
    保存 B 矩阵的热力图到文件。
    参数:
        B (np.ndarray): 基矩阵，形状为 (n, n)
        iteration (int): 当前迭代次数
        output_dir (str): 图像保存目录
    """
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

# ============ 主算法：迭代构造晶格基 ============

def iterative_lattice_construction(n,
                                   T=10000,
                                   Tr=1000,
                                   mu0=0.01,
                                   nu=500.0,
                                   tqdm_update_freq=1000,
                                   save_plots=False,
                                   output_dir='B_plots',
                                   theta_r_values=None,
                                   method='Monte Carlo',
                                   num_samples=100000):
    """
    迭代构造晶格基，并记录 Theta 图像所需的数据。
    参数:
        n (int): 维度
        T (int): 主循环次数
        Tr (int): 每 Tr 次迭代进行一次 RED + ORTH
        mu0 (float): 初始步长
        nu (float): 步长衰减因子
        tqdm_update_freq (int): 每多少次迭代更新一次进度条
        save_plots (bool): 是否保存 B 矩阵的图像
        output_dir (str): 图像保存目录
        theta_r_values (list or np.ndarray): 要计算的半径值列表
        method (str): 计算 Theta 的方法，'Exact' 或 'Monte Carlo'
        num_samples (int): 蒙特卡洛采样次数（仅当 method='Monte Carlo' 时有效）
    返回:
        B (np.ndarray): 迭代后得到的生成矩阵
        theta_history (dict): 记录每次 RED 后的 N(B, r) 值，按 r 分组
    """
    norms_history = []
    theta_history = {r: [] for r in theta_r_values} if theta_r_values is not None else {}

    # 1. B <- ORTH( RED( GRAN(n,n) ) )
    B_init = GRAN(n, n)
    B_red = RED(B_init)
    B = ORTH(B_red)

    # 2. B <- V^(-1/n) * B, 其中 V = ∏ B[i,i]
    diag_prod = np.prod(np.diag(B))
    alpha = diag_prod ** (-1.0 / n)
    B = alpha * B

    # 记录初始的 Theta
    if theta_r_values is not None:
        for r in theta_r_values:
            if method == 'Exact' and n <= 5:  # 设定 n 的上限，防止高维度的确切计数
                theta = count_lattice_points_exact(B, r)
            elif method == 'Monte Carlo':
                theta = count_lattice_points_monte_carlo(B, r, num_samples=num_samples)
            else:
                theta = None  # 不支持的方法
            theta_history[r].append(theta)

    # 主循环
    with tqdm(total=T, desc="迭代进度") as pbar:
        for t in range(T):
            # 更新步长 μ
            if T > 1:
                mu = mu0 * (nu ** (-float(t) / (T - 1)))
            else:
                mu = mu0

            # 生成随机向量 z
            z = URAN(n)

            # 最近晶格点
            zB = z @ B
            u_hat = CLP(B, zB)
            y = z - u_hat

            # 计算误差向量 e
            e = y @ B

            # 更新 B 矩阵
            e_norm_sq = np.linalg.norm(e) ** 2
            for i_idx in range(n):
                for j_idx in range(i_idx):
                    B[i_idx, j_idx] -= mu * y[i_idx] * e[j_idx]
                # 更新对角元素
                bii = B[i_idx, i_idx]
                if bii != 0:
                    B[i_idx, i_idx] = bii - mu * (y[i_idx]*e[i_idx] - e_norm_sq / (n * bii))
                else:
                    # 避免除以零
                    B[i_idx, i_idx] = bii - mu * y[i_idx] * e[i_idx]

            # 每 Tr 步进行一次 RED 和 ORTH
            if (t % Tr) == (Tr - 1):
                B = RED(B)
                B = ORTH(B)
                # 再次缩放
                diag_prod = np.prod(np.diag(B))
                alpha = diag_prod ** (-1.0 / n)
                B = alpha * B

                # 记录当前 Theta
                if theta_r_values is not None:
                    for r in theta_r_values:
                        if method == 'Exact' and n <= 5:
                            theta = count_lattice_points_exact(B, r)
                        elif method == 'Monte Carlo':
                            theta = count_lattice_points_monte_carlo(B, r, num_samples=num_samples)
                        else:
                            theta = None
                        theta_history[r].append(theta)

                # 保存 B 矩阵的图像
                if save_plots:
                    iteration = t + 1
                    save_B_matrix_plot(B, iteration, output_dir)

            # 更新进度条
            if (t + 1) % tqdm_update_freq == 0:
                pbar.update(tqdm_update_freq)

    return B, theta_history

# ============ 测试/示例 ============

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    # 参数设置
    n = 3              # 维度（低维度以便使用确切计数）
    T = int(1e5)          # 迭代次数
    Tr = 1000          # 每 Tr 次迭代进行一次 RED + ORTH
    mu0 = 0.01         # 初始步长
    nu = 500.0         # 步长衰减因子
    tqdm_update_freq = 1000  # 进度条更新频率
    save_plots = False        # 是否保存 B 矩阵图像
    output_dir = 'B_plots'    # 图像保存目录

    # Theta 图像所需的半径值
    theta_r_values = np.linspace(0, 3, 10)  # 从 0 到 3 的 10 个 r 值

    # 选择计算方法：'Exact' 或 'Monte Carlo'
    method = 'Exact' if n <=5 else 'Monte Carlo'
    num_samples = 100000  # 蒙特卡洛采样次数

    # 执行迭代晶格构造
    B_final, theta_history = iterative_lattice_construction(
        n=n,
        T=T,
        Tr=Tr,
        mu0=mu0,
        nu=nu,
        tqdm_update_freq=tqdm_update_freq,
        save_plots=save_plots,
        output_dir=output_dir,
        theta_r_values=theta_r_values,
        method=method,
        num_samples=num_samples
    )

    print("最终得到的 B 矩阵：")
    print(B_final)

    # 绘制最终 B 矩阵的热力图
    plot_B_matrix(B_final)

    # 绘制 Theta 图像
    # 为了展示多个 r 值的 Theta 图像，绘制每个 r 的 N(B, r) 随迭代次数的变化
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, T +1)
    for r in theta_r_values:
        theta_values = theta_history[r]
        red_iterations = np.arange(len(theta_values)) * Tr
        plt.step(red_iterations, theta_values, where='post', label=f'r={r:.2f}')
    plt.xlabel('迭代次数')
    plt.ylabel('$N(B, r)$')
    plt.title('Theta 图像的演化')
    plt.legend()
    plt.grid(True)
    plt.show()
    # save the plot
    plt.savefig('Theta_evolution.png')

    # 最终 Theta 图像：N(B, r) 与 r^2 的关系
    # 选择最后一次记录的 N(B, r) 值
    final_theta = [theta_history[r][-1] for r in theta_r_values]
    final_r_squared = theta_r_values**2

    plt.figure(figsize=(8,6))
    plt.plot(final_r_squared, final_theta, marker='o', linestyle='-', label='最终 Theta')
    plt.xlabel('$r^2$')
    plt.ylabel('$N(B, r)$')
    plt.title('最终的 Theta 图像')
    plt.legend()
    plt.grid(True)
    plt.show()
    # save the plot
    plt.savefig('Final_Theta.png')