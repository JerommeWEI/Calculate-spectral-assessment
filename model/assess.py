import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

class FPICoatingEvaluator:
    def __init__(self, wavelengths, sensing_matrix):
        """
        初始化评估器
        :param wavelengths: 波长数组 (1D array, 例如 400 到 950)
        :param sensing_matrix: 传感矩阵 Phi (N_states x M_wavelengths)
        """
        self.wl = wavelengths
        self.Phi = sensing_matrix
        self.N_states, self.M_wl = self.Phi.shape
        
    def analyze_matrix(self):
        """1. 评估传感矩阵的数学适定性：相关性与奇异值衰减"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # a. 绘制传感矩阵热力图
        im1 = axes[0].imshow(self.Phi, aspect='auto', cmap='viridis', 
                             extent=[self.wl[0], self.wl[-1], self.N_states, 1])
        axes[0].set_title('Sensing Matrix $\Phi$ (T($\lambda$, d))')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('FPI State (Cavity Length Index)')
        fig.colorbar(im1, ax=axes[0])
        
        # b. 互相关矩阵 (Cross-Correlation)
        # 计算列与列（波长与波长）之间的相关性
        corr_matrix = np.corrcoef(self.Phi, rowvar=False) 
        im2 = axes[1].imshow(corr_matrix, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1,
                             extent=[self.wl[0], self.wl[-1], self.wl[-1], self.wl[0]])
        axes[1].set_title('Wavelength Cross-Correlation')
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('Wavelength (nm)')
        fig.colorbar(im2, ax=axes[1])
        
        # c. 奇异值分解 (SVD)
        U, S, V = np.linalg.svd(self.Phi, full_matrices=False)
        axes[2].plot(np.log10(S / S[0] + 1e-12), 'b.-')
        axes[2].set_title('Singular Value Decay (Log Scale)')
        axes[2].set_xlabel('Index')
        axes[2].set_ylabel('Log10(S_i / S_0)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    def add_noise(self, y, snr_db):
        """为测量信号添加高斯白噪声"""
        signal_power = np.mean(y**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(*y.shape)
        return y + noise

    def simulate_reconstruction(self, signal_type='sparse', snr_db=30):
        """2. 闭环仿真：高斯单峰（稀疏）或连续光谱（稠密）重建"""
        x_true = np.zeros(self.M_wl)
        
        # 生成测试信号
        if signal_type == 'sparse':
            # 单点高斯激光 (窄带)
            center_idx = self.M_wl // 2
            x_true = np.exp(-0.5 * ((np.arange(self.M_wl) - center_idx) / 5)**2)
            model = Lasso(alpha=1e-4, max_iter=10000) # 稀疏信号使用 L1 正则化 (Lasso)
            title = 'Sparse Signal (Laser Peak) Reconstruction'
        else:
            # 连续光谱 (宽带，如卤素灯或反射包络)
            x_true = np.exp(-0.5 * ((np.arange(self.M_wl) - self.M_wl//3) / 80)**2) + \
                     0.5 * np.exp(-0.5 * ((np.arange(self.M_wl) - 2*self.M_wl//3) / 100)**2)
            model = Ridge(alpha=1e-2) # 稠密连续信号使用 L2 正则化 (Ridge / Tikhonov)
            title = 'Continuous Spectrum Reconstruction'
            
        # 前向传播：物理测量 + 噪声
        y_clean = self.Phi @ x_true
        y_noisy = self.add_noise(y_clean, snr_db)
        
        # 逆向求解：算法重构
        model.fit(self.Phi, y_noisy)
        x_recon = model.coef_
        
        # 结果评价
        rmse = np.sqrt(mean_squared_error(x_true, x_recon))
        
        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(self.wl, x_true, 'k-', lw=2, label='Ground Truth')
        plt.plot(self.wl, x_recon, 'r--', lw=2, label=f'Reconstructed (RMSE={rmse:.4f})')
        plt.title(f"{title} @ SNR={snr_db}dB")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        plt.show()

# ==========================================
# 测试与运行演示 (如何使用上述类)
# ==========================================
if __name__ == "__main__":
    # 1. 设置波长范围 400 ~ 950 nm，间隔 1nm
    wl = np.linspace(400, 950, 551)
    
    # 2. 【重要】这里你需要替换为你从 Essential Macleod 导出的数据
    # 下面这段代码只是用简化的 Airy 公式生成一个虚拟的响应矩阵进行演示
    N_cavity_states = 64 # 假设 FPI 扫描了 64 个不同的腔长状态
    cavity_lengths = np.linspace(500, 1500, N_cavity_states) 
    Phi_mock = np.zeros((N_cavity_states, len(wl)))
    
    R = 0.85 # 假设膜系反射率为 0.85 (实际介质膜的 R 是随波长变化的，你的导入数据会自动包含此特性)
    for i, d in enumerate(cavity_lengths):
        # 简化的透射率模型
        delta = (4 * np.pi * d) / wl
        Phi_mock[i, :] = ((1 - R)**2) / (1 + R**2 - 2 * R * np.cos(delta))
        
    # === 开始评估 ===
    # 实例化评估器
    evaluator = FPICoatingEvaluator(wl, Phi_mock)
    
    # 第一步：分析矩阵特性 (肉眼判断膜系设计是否合理)
    print("正在进行矩阵适定性分析...")
    evaluator.analyze_matrix()
    
    # 第二步：测试单点高斯激光的重建能力 (SNR = 30dB)
    print("正在进行单峰激光信号重建测试...")
    evaluator.simulate_reconstruction(signal_type='sparse', snr_db=30)
    
    # 第三步：测试宽带连续光谱的重建能力 (SNR = 40dB)
    print("正在进行连续光谱信号重建测试...")
    evaluator.simulate_reconstruction(signal_type='continuous', snr_db=40)