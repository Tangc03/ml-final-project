import numpy as np
from utils.LLL import LLL_reduction
from utils.new_search_fixed import search_nearest_point, sign
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.special import gamma  # 导入伽玛函数

from utils.compute_nsm import compute_nsm

# 设置全局字体为支持中文的字体（如 SimHei 或 Microsoft YaHei）
plt.rcParams['font.family'] = 'Microsoft YaHei' # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

def plot_B_matrix(B, output_file='B_matrix.png'):
    plt.figure(figsize=(8, 6))
    plt.imshow(B, cmap='viridis', aspect='auto')
    plt.colorbar(label='值')
    plt.title('最终得到的 B 矩阵')
    plt.xlabel('列索引')
    plt.ylabel('行索引')
    plt.savefig(output_file)
    # plt.show()

def plot_norms_history(norms_history, output_file='norms_history.png'):
    norms_history = np.array(norms_history)
    plt.figure(figsize=(10, 6))
    for i in range(norms_history.shape[1]):
        plt.plot(norms_history[:, i], label=f'行 {i+1} 的范数')
    plt.xlabel('RED+ORTH 迭代次数')
    plt.ylabel('行向量范数')
    plt.title('B 矩阵行向量范数的变化')
    plt.legend()
    plt.savefig(output_file)
    # plt.show()

# ============ Theta 图像绘制相关函数 ============

def count_lattice_points(B, r):
    """
    计算在半径 r 内的晶格点数量 N(B, r)
    
    参数:
        B: 生成矩阵 (n x n)
        r: 半径
    
    返回:
        count: 晶格点数量
    """
    n = B.shape[0]
    count = 0
    r_squared = r**2

    # 递归函数生成 u 向量
    def recurse(dim, current_u, current_norm_sq):
        nonlocal count
        if dim == n:
            if current_norm_sq <= r_squared:
                count += 1
            return
        # 估计当前维度的可能范围
        # ||uB||^2 = sum_{i=1}^n (sum_{j=1}^n u_j B_{j,i})^2 <= r^2
        # 这里进行简单的限制，避免过多递归
        # 计算每个维度 u_j 的范围
        # 这里假设 B 的列已被正交化，可以进一步优化
        for u_j in range(-int(r) - 1, int(r) + 2):
            new_norm_sq = current_norm_sq + (u_j * B[dim, dim])**2
            if new_norm_sq <= r_squared:
                recurse(dim + 1, current_u + [u_j], new_norm_sq)

    recurse(0, [], 0)
    return count

def count_lattice_points_monte_carlo(B, r, num_samples=100000):
    """
    使用蒙特卡洛方法估计在半径 r 内的晶格点数量 N(B, r)
    
    参数:
        B: 生成矩阵 (n x n)
        r: 半径
        num_samples: 采样次数
    
    返回:
        estimated_count: 估计的晶格点数量
    """
    n = B.shape[0]
    count_within_r = 0

    # 计算每个维度的采样范围 U
    col_norms = np.linalg.norm(B, axis=0)
    min_col_norm = np.min(col_norms)
    if min_col_norm == 0:
        raise ValueError("生成矩阵 B 存在列范数为零的情况，无法进行蒙特卡洛估计。")
    U = int(np.ceil(r / min_col_norm)) + 1

    # 随机采样整数向量 u ∈ [-U, U]^n
    u_samples = np.random.randint(-U, U+1, size=(num_samples, n))

    # 计算对应的晶格点
    lattice_points = u_samples @ B.T  # shape: (num_samples, n)

    # 计算范数
    norms = np.linalg.norm(lattice_points, axis=1)

    # 统计在半径 r 内的点数
    count_within_r = np.sum(norms <= r)

    # 估计体积 V
    # n 维球体的体积 V_n(r) = (π^(n/2) / Γ(n/2 + 1)) * r^n
    V_n_r = (np.pi ** (n / 2)) / gamma(n / 2 + 1) * (r ** n)
    V_total = (2 * U) ** n

    # 估计 N(B, r) = (count_within_r / num_samples) * (V_total / V_n_r)
    estimated_count = (count_within_r / num_samples) * (V_total / V_n_r)

    return estimated_count

def plot_theta_image(B, r_max=10.0, r_step=0.1, output_file='theta_image.png', method='exact', num_samples=100000):
    """
    绘制 Theta 图像 N(B, r) 与 r^2 的关系
    
    参数:
        B: 生成矩阵 (n x n)
        r_max: 最大半径
        r_step: 半径步长
        output_file: 图像保存路径
        method: 计算方法，'exact' 或 'monte_carlo'
        num_samples: 蒙特卡洛方法的采样次数（仅在 method='monte_carlo' 时使用）
    """
    rs = np.arange(0, r_max + r_step, r_step)
    N_values = []
    
    print(f"开始计算 N(B, r) 使用方法: {method}...")
    for r in tqdm(rs, desc="计算 N(B, r)"):
        if method == 'exact':
            N = count_lattice_points(B, r)
        elif method == 'monte_carlo':
            N = count_lattice_points_monte_carlo(B, r, num_samples=num_samples)
        else:
            raise ValueError("method 参数必须为 'exact' 或 'monte_carlo'")
        N_values.append(N)
    
    rs_squared = rs**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(rs_squared, N_values, drawstyle='steps-post')
    plt.xlabel('$r^2$')
    plt.ylabel('$N(B, r)$')
    plt.title('Theta 图像 $N(B, r)$ vs $r^2$')
    plt.grid(True)
    plt.savefig(output_file)
    # plt.show()
    print(f"Theta 图像已保存至 {output_file}")

# ============ 主算法：迭代构造晶格基 ============

def iterative_lattice_construction(n,
                                   T=10,
                                   Tr=3,
                                   mu0=1.0,
                                   nu=2.0,
                                   tqdm_update_freq=1000,
                                   save_plots=False,
                                   output_dir='B_plots',
                                   compute_theta=False,
                                   theta_r_max=10.0,
                                   theta_r_step=0.1,
                                   theta_output_file='theta_image.png',
                                   theta_method='exact',
                                   theta_num_samples=100000):
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
      compute_theta: 是否计算并绘制 Theta 图像
      theta_r_max: Theta 图像的最大半径
      theta_r_step: Theta 图像的半径步长
      theta_output_file: Theta 图像的保存路径
      theta_method: 计算 Theta 图像的方法，'exact' 或 'monte_carlo'
      theta_num_samples: 蒙特卡洛方法的采样次数
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
            # 5. 更新 μ = mu0 * nu^(-t/(T-1))
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

    # 绘制最终的 B 矩阵和范数历史
    if compute_theta:
        plot_theta_image(B, r_max=theta_r_max, r_step=theta_r_step, output_file=theta_output_file,
                        method=theta_method, num_samples=theta_num_samples)

    return B, norms_history

# ============ 测试/示例 ============

if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子便于演示
    # n = 10             # 维度
    T = int(1e4)       # 迭代次数，示例中使用较小的值
    Tr = 100           # 每隔多少步做一次 RED+ORTH
    mu0 = 0.005
    nu = 200.0

    # # 启用保存图像和 Theta 图像绘制
    # B_final, norms_history = iterative_lattice_construction(
    #     n, T, Tr, mu0, nu, save_plots=True, output_dir='B_plots',
    #     compute_theta=True, theta_r_max=1.5, theta_r_step=0.5,
    #     theta_output_file='theta_image.png',
    #     # theta_method='monte_carlo',  # 选择使用蒙特卡洛方法
    #     theta_num_samples=100000      # 设定采样次数
    # )
    # print("最终得到的 B 矩阵：")
    # print(B_final)
    # plot_B_matrix(B_final)
    # plot_norms_history(norms_history)
    
    # creating folder for saving images
    # only work on Windows machine
    os.system('rd /s /q theta_images')
    os.makedirs('theta_images')
    os.system('rd /s /q B_matrixes')
    os.makedirs('B_matrixes')
    os.system('rd /s /q norms_histories')
    os.makedirs('norms_histories')

    nsm=[]
    n_list=[10,11]
    #n_list = [10, 11, 12, 13, 14, 15, 16]
    for n in n_list:
        print(f"开始计算维度为 {n} 的晶格基...")
        # 启用保存图像和 Theta 图像绘制
        B_final, norms_history = iterative_lattice_construction(
            n, T, Tr, mu0, nu, save_plots=True, output_dir='B_plots/n_{}'.format(n),
            compute_theta=True, theta_r_max=1.8, theta_r_step=0.3,
            theta_output_file='theta_images/theta_image_n_{}.png'.format(n),
            # theta_method='monte_carlo',  # 选择使用蒙特卡洛方法
            # theta_num_samples=100000      # 设定采样次数
        )
        print("最终得到的 B 矩阵：")
        print(B_final)
        nsm.append(compute_nsm(B_final))
        #calcualte nsm and save to .txt document

        plot_B_matrix(B_final, output_file='B_matrixes/B_matrix_n_{}.png'.format(n))
        plot_norms_history(norms_history, output_file='norms_histories/norms_history_n_{}.png'.format(n))
    
    with open('nsm.txt','w') as f:
        for i in nsm:
            f.write(f'{str(i)}\n')
    print("FINISHED")