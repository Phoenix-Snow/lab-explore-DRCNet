import os
import seaborn as sns
from scipy import stats

from sklearn.manifold import TSNE as sk_tsne
from openTSNE import TSNE as open_tsne
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline

from Tools.Tools import safe_name, ged_func

# 可在此处统一设置，或函数调用时一对一地设置
_perplexity = 10


# t-SNE 图
def draw_tsne_by_ss(features, labels, mode, path, perplexity=_perplexity):
    """
    样本过多（尤其是时序）, 采用 ShuffleSplit 划分出部分样本来做图, 此方法不能兼顾 label 的均衡问题。（回归检测：用于确定时序等连续性情况）
    """
    tsne_tool = 0  # sklearn
    # tsne_tool = 1  # openTSNE

    print(f'->->-> Drawing TSNE (task: {mode}):')
    features = features.cpu().detach().numpy().reshape(-1, features.shape[-1])  # 针对f做 - 全参数
    labels = labels.cpu().detach().numpy().reshape(-1, labels.shape[-1])  # 针对f做 - 全结果 -> 切出第一个输出做图

    # 防止StratifiedShuffleSplit报ValueError
    # k = 10
    # pseudo_label = KMeans(n_clusters=k, random_state=0).fit_predict(labels)
    # label_counter = Counter(pseudo_label)
    # mask = np.array([label_counter[l] >= 2 for l in labels], dtype=bool)
    # features, labels = features[mask], labels[mask]

    if len(features) > 10000:  # 如果样本数量过大
        # ss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
        ss = ShuffleSplit(n_splits=1, train_size=10000, random_state=42)
        for idx, _ in ss.split(features, labels):
            features, labels = features[idx], labels[idx]
        print(f'   Organize dataset to {len(features)} for TSNE.')

    if features.shape[-1] >= 2:
        print('   1- Reducing dimensions ...')
        if tsne_tool == 0:
            tsne = sk_tsne(n_components=2, random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit_transform(features)  # sklearn
        else:
            tsne = open_tsne(n_components=2, random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit(features)  # openTSNE
        print('    > Done!')
    else:  # 如果特征数为1的话
        print('   1- Reducing dimensions ...')
        if tsne_tool == 0:
            tsne = sk_tsne(n_components=2, init='random', random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit_transform(features)  # sklearn
        else:
            tsne = open_tsne(n_components=2, initialization='random', random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit(features)  # openTSNE
        print('    > Done!')

    # 可视化
    print('   2- Drawing ...')
    plt.scatter(low_dim_vectors[:, 0], low_dim_vectors[:, 1], c=labels, cmap='tab10')
    plt.title(f"t-SNE visualization of latent vectors ({mode})")
    save_path = os.path.join(path, f'tsne_{safe_name(mode)}.png')
    plt.savefig(save_path)  # 文件名中特殊字符替换掉，空格去掉
    plt.close()

    print('    > Done!')
    print(f'(task: {mode}) : Over <-<-<-')


def draw_tsne_by_sss(features, labels, mode, path, perplexity=_perplexity):
    """
    样本过多（尤其是时序）, 采用 StratifiedShuffleSplit 划分出部分样本来做图, 此方法可以根据 label 均衡提取样本。
    （类别检测：用于确定季节、地理等固定标签的聚类情况）
    注意：对于回归检测, 因为 label 是连续的，划分 label 容易出现一种 label 只有一个样本的情况。
    """
    tsne_tool = 0  # sklearn
    # tsne_tool = 1  # openTSNE

    print(f'->->-> Drawing TSNE (task: {mode}):')
    features = features.cpu().detach().numpy().reshape(-1, features.shape[-1])  # 针对f做 - 全参数
    labels = labels.cpu().detach().numpy().reshape(-1, labels.shape[-1])  # 针对f做 - 全结果 -> 切出第一个输出做图

    # 防止StratifiedShuffleSplit报ValueError
    # k = 10
    # pseudo_label = KMeans(n_clusters=k, random_state=0).fit_predict(labels)
    # label_counter = Counter(pseudo_label)
    # mask = np.array([label_counter[l] >= 2 for l in labels], dtype=bool)
    # features, labels = features[mask], labels[mask]

    if len(features) > 10000:  # 如果样本数量过大
        sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
        for idx, _ in sss.split(features, labels):
            features, labels = features[idx], labels[idx]
        print(f'   Organize dataset to {len(features)} for TSNE.')

    if features.shape[-1] >= 2:
        print('   1- Reducing dimensions ...')
        if tsne_tool == 0:
            tsne = sk_tsne(n_components=2, random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit_transform(features)  # sklearn
        else:
            tsne = open_tsne(n_components=2, random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit(features)  # openTSNE
        print('    > Done!')
    else:  # 如果特征数为1的话
        print('   1- Reducing dimensions ...')
        if tsne_tool == 0:
            tsne = sk_tsne(n_components=2, init='random', random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit_transform(features)  # sklearn
        else:
            tsne = open_tsne(n_components=2, initialization='random', random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit(features)  # openTSNE
        print('    > Done!')

    # 可视化
    print('   2- Drawing ...')
    plt.scatter(low_dim_vectors[:, 0], low_dim_vectors[:, 1], c=labels, cmap='tab10')
    plt.title(f"t-SNE visualization of latent vectors ({mode})")
    save_path = os.path.join(path, f'tsne_{safe_name(mode)}.png')
    plt.savefig(save_path)  # 文件名中特殊字符替换掉，空格去掉
    plt.close()

    print('    > Done!')
    print(f'(task: {mode}) : Over <-<-<-')


def draw_tsne_by_feature_sss(features, labels, mode, path, perplexity=_perplexity):
    """
    样本过多（尤其是时序）, 采用 StratifiedShuffleSplit 划分出部分样本来做图, 此方法可以根据 feature 均衡提取样本。
    （回归检测: 对隐向量做的划分, 看隐向量的聚类情况, 因为隐向量是连续的, 往往有概率会出问题）
    """
    tsne_tool = 0  # sklearn
    # tsne_tool = 1  # openTSNE

    print(f'->->-> Drawing TSNE (task: {mode}):')
    features = features.cpu().detach().numpy().reshape(-1, features.shape[-1])  # 针对f做 - 全参数
    labels = labels.cpu().detach().numpy().reshape(-1, labels.shape[-1])  # 针对f做 - 全结果 -> 切出第一个输出做图

    # 防止StratifiedShuffleSplit报ValueError
    # k = 10
    # pseudo_label = KMeans(n_clusters=k, random_state=0).fit_predict(labels)
    # label_counter = Counter(pseudo_label)
    # mask = np.array([label_counter[l] >= 2 for l in labels], dtype=bool)
    # features, labels = features[mask], labels[mask]

    if len(features) > 10000:  # 如果样本数量过大
        sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
        for idx, _ in sss.split(labels, features):
            features, labels = features[idx], labels[idx]
        print(f'   Organize dataset to {len(features)} for TSNE.')

    if features.shape[-1] >= 2:
        print('   1- Reducing dimensions ...')
        if tsne_tool == 0:
            tsne = sk_tsne(n_components=2, random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit_transform(features)  # sklearn
        else:
            tsne = open_tsne(n_components=2, random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit(features)  # openTSNE
        print('    > Done!')
    else:  # 如果特征数为1的话
        print('   1- Reducing dimensions ...')
        if tsne_tool == 0:
            tsne = sk_tsne(n_components=2, init='random', random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit_transform(features)  # sklearn
        else:
            tsne = open_tsne(n_components=2, initialization='random', random_state=42, perplexity=perplexity)
            low_dim_vectors = tsne.fit(features)  # openTSNE
        print('    > Done!')

    # 可视化
    print('   2- Drawing ...')
    plt.scatter(low_dim_vectors[:, 0], low_dim_vectors[:, 1], c=labels, cmap='tab10')
    plt.title(f"t-SNE visualization of latent vectors ({mode})")
    save_path = os.path.join(path, f'tsne_{safe_name(mode)}.png')
    plt.savefig(save_path)  # 文件名中特殊字符替换掉，空格去掉
    plt.close()

    print('    > Done!')
    print(f'(task: {mode}) : Over <-<-<-')


# 三大误差图
def draw_residuals(y_true, y_pred, mode, path):
    """
    回归任务，采用 ShuffleSplit 划分，绘制残差、Q-Q、KDE图
    """
    print(f'->->-> Drawing PLOT BTF-RESIDUALS (task: {mode}): ...')
    assert y_true.shape == y_pred.shape, "y_true 与 y_pred 形状必须一致"
    os.makedirs(path, exist_ok=True)

    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred_ss = y_pred.copy()
    y_true_ss = y_true.copy()

    if len(y_true) > 1000:  # 样本太多，对于部分误差图计算太困难，创建缩减版副本
        sss = ShuffleSplit(n_splits=1, train_size=1000, random_state=42)
        for idx, _ in sss.split(y_pred, y_true):
            y_pred_ss, y_true_ss = y_pred[idx], y_true[idx]
        print(f'   Organize {len(y_pred)} dataset to {len(y_pred_ss)} for part of plot-btf-residuals.')
    else:
        pass

    res_1d_ss = (y_true_ss - y_pred_ss).ravel()
    fitted_1d_ss = y_pred_ss.ravel()
    res_1d = (y_true - y_pred).ravel()
    # fitted_1d = y_pred.ravel()

    del y_true, y_pred, y_true_ss, y_pred_ss  # 算完清除不必要数据，释放内存

    base_name = os.path.join(path, safe_name(mode))

    # ---------- 图 1：残差 vs 拟合 ----------
    # print('   1- Drawing (Residuals vs Fitted) ...')
    # fig1, ax1 = plt.subplots()
    # sns.residplot(x=fitted_1d, y=res_1d, lowess=True,
    #               scatter_kws={'alpha': 0.4, 's': 10}, ax2=ax1)
    # ax1.axhline(0, ls='--', c='grey')
    # ax1.set_title('Residuals vs Fitted')
    # ax1.set_xlabel('Fitted (predicted) values')
    # ax1.set_ylabel('Residuals')
    # fig1.savefig(f'{base_name}_1_residual_vs_fitted.png', dpi=600, bbox_inches='tight')
    # plt.close(fig1)
    print('   1- Drawing (Residuals vs Fitted) ...')
    try:
        fig1, ax1 = plt.subplots()
        sns.residplot(x=fitted_1d_ss, y=res_1d_ss,
                      scatter_kws={'alpha': 0.4, 's': 10},
                      line_kws={},  # 把默认的置信区间线也关掉
                      lowess=False, ax=ax1)
        lowess_y = lowess(res_1d_ss, fitted_1d_ss, frac=0.2, return_sorted=True)  # frac 控制平滑度
        x_smooth, y_smooth = lowess_y[:, 0], lowess_y[:, 1]
        ax1.plot(x_smooth, y_smooth, color='green', linewidth=1.5)
        ax1.axhline(0, ls='--', c='grey')
        ax1.set_title('Residuals vs Fitted')
        ax1.set_xlabel('Fitted (predicted) values')
        ax1.set_ylabel('Residuals')

        fig1.savefig(f'{base_name}_1_residual_vs_fitted.png',
                     dpi=600, bbox_inches='tight')
        plt.close(fig1)
        print('    > Residuals vs Fitted -> Done!')
    except Exception as e:
        print(f'    > Residuals vs Fitted -> Fail:\n'
              f'      {e}')

    # ---------- 图 2：直方图 + KDE ----------
    print('   2- Drawing (Histogram of residuals) ...')
    try:
        fig2, ax2 = plt.subplots()
        sns.histplot(res_1d, kde=True, bins=min(50, int(np.sqrt(res_1d.size))), ax=ax2)
        ax2.axvline(0, ls='--', c='red')
        ax2.set_title('Histogram of residuals')
        fig2.savefig(f'{base_name}_2_histogram.png', dpi=600, bbox_inches='tight')
        plt.close(fig2)
        print('    > Histogram of residuals -> Done!')
    except Exception as e:
        print(f'    > Histogram of residuals -> Fail:\n'
              f'      {e}')

    # ---------- 图 3：Q-Q 图 ----------
    print('   3- Drawing (Q-Q) ...')
    # ---------- norm ----------
    try:
        fig3, ax3 = plt.subplots()
        stats.probplot(res_1d, dist="norm", plot=ax3)  # nomal方法
        ax3.set_title('Normal Q-Q')
        fig3.savefig(f'{base_name}_3_qq_norm.png', dpi=600, bbox_inches='tight')
        plt.close(fig3)
        print('    > Normal Q-Q -> Done!')
    except Exception as e:
        print(f'    > Normal Q-Q -> Fail: \n'
              f'      {e}')

    # 不使用lognorm和gamma，因为这里画的是误差的Q-Q图，如果连负值和零值都不能有就一定存在问题

    # a_gm, loc_gm, scale_gm = stats.gamma.fit(res_1d, floc=0)
    # fig4, ax4 = plt.subplots()
    # stats.probplot(res_1d, dist=stats.gamma(a_gm, loc=loc_gm, scale=scale_gm), plot=ax4)  # 自定义gamma方法
    # ax4.set_title('Gamma Q-Q')
    # fig4.savefig(f'{base_name}_3_qq_gamma.png', dpi=600, bbox_inches='tight')
    # plt.close(fig4)

    # ---------- laplace 拉普拉斯 ----------
    try:
        fig5, ax5 = plt.subplots()
        stats.probplot(res_1d, dist="laplace", plot=ax5)  # laplace方法
        ax5.set_title('Laplace Q-Q')
        fig5.savefig(f'{base_name}_3_qq_laplace.png', dpi=600, bbox_inches='tight')
        plt.close(fig5)
        print('    > Laplace Q-Q -> Done!')
    except Exception as e:
        print(f'    > Laplace Q-Q -> Fail: \n'
              f'      {e}')

    # ---------- t ----------
    try:
        try:
            df, loc_t, scale_t = stats.t.fit(res_1d)
            dist_t = stats.t(df, loc=loc_t, scale=scale_t)
        except Exception as e_:
            print(f'    > [t Q-Q] WARNING: {e_}'
                  f'    > Fit t dist_t -> stats.t(3)\n')
            dist_t = stats.t(3)  # 默认自由度为3
        fig6, ax6 = plt.subplots()
        stats.probplot(res_1d, dist=dist_t, plot=ax6)  # 自定义t方法
        ax6.set_title('t-Distribution Q-Q')
        fig6.savefig(f'{base_name}_3_qq_t.png', dpi=600, bbox_inches='tight')
        plt.close(fig6)
        print('    > t Q-Q -> Done!')
    except Exception as e:
        print(f'    > t Q-Q -> Fail: \n'
              f'      {e}')

    # ---------- ged ----------
    try:
        fig7, ax7 = plt.subplots()
        stats.probplot(res_1d, dist=ged_func(res_1d), plot=ax7)
        ax7.set_title('GED (gennorm) Q-Q')
        fig7.savefig(f'{base_name}_3_qq_ged.png', dpi=600, bbox_inches='tight')
        plt.close(fig7)
        print('    > GED Q-Q -> Done!')
    except Exception as e:
        print(f'    > GED Q-Q -> Fail: \n'
              f'      {e}')

    # # ---------- 图 3：P-P 图 ----------
    # print('   4- Drawing (P-P) ...')
    # # # 核密度估计（KDE）图
    # # try:
    # #     # 绘制KDE图
    # #     fig8, ax8 = plt.subplots()
    # #     sns.kdeplot(res_1d, fill=True, ax=ax8)
    # #     ax8.set_title('Kernel Density Estimate (KDE)')
    # #     fig8.savefig(f'{base_name}_4_kde.png', dpi=600, bbox_inches='tight')
    # #     plt.close(fig8)
    # #     print('    > KDE -> Done!')
    # # except Exception as e:
    # #     print(f'    > KDE -> Fail: \n'
    # #           f'      {e}')
    # kde = stats.gaussian_kde(res_1d_ss)
    # sorted_data = np.sort(res_1d_ss)
    # sorted_pdf = kde(sorted_data)
    # theoretical_cdf = np.array([kde.integrate_box_1d(-np.inf, x) for x in sorted_data])
    # # 分位数-秩图
    # try:
    #     # 直线＝单调对应没问题；曲线弯了＝模型压根不适合这批数据
    #     fig9, ax9 = plt.subplots()
    #     ax9.plot([0, 1], [0, 1], 'r--')
    #     ax9.scatter(theoretical_cdf, np.linspace(0, 1, len(sorted_data)), s=5, c='b')
    #     ax9.set_title('P-P Plot')
    #     ax9.set_xlabel('Theoretical CDF')
    #     ax9.set_ylabel('Empirical CDF')
    #     fig9.savefig(f'{base_name}_4_pp_rp.png', dpi=600, bbox_inches='tight')
    #     plt.close(fig9)
    #     print('    > P-P(rankit plot) -> Done!')
    # except Exception as e:
    #     print(f'    > P-P(rankit plot) -> Fail: \n'
    #           f'      {e}')
    # # 密度–CDF 双轴图
    # try:
    #     fig10, ax10 = plt.subplots()
    #     ax10_twin = ax10.twinx()  # 右侧 Y 轴
    #     # 左轴：KDE 密度曲线
    #     ax10.plot(sorted_data, sorted_pdf, 'b-', lw=1.5, label='KDE density')
    #     ax10.set_ylabel('Density', color='b')
    #     ax10.tick_params(axis='y', labelcolor='b')
    #     # 右轴：理论 CDF
    #     ax10_twin.plot(sorted_data, theoretical_cdf, 'r-', lw=1.5, label='KDE CDF')
    #     ax10_twin.set_ylabel('Cumulative probability', color='r')
    #     ax10_twin.tick_params(axis='y', labelcolor='r')
    #     ax10_twin.set_ylim(0, 1)
    #     # 公共 X 轴
    #     ax10.set_xlabel('Value')
    #     ax10.set_title('Density–CDF Dual-Axis Plot')
    #     # 保存
    #     fig10.savefig(f'{base_name}_5_pp_density_cdf.png', dpi=600, bbox_inches='tight')
    #     plt.close(fig10)
    #     print('    > P-P(Density–CDF) -> Done!')
    # except Exception as e:
    #     print(f'    > P-P(Density–CDF) -> Fail: \n'
    #           f'      {e}')

    print(f"  -- four residual plots type (9 imgs) saved to -> {base_name}_*.png")

    print(f'(task: {mode}) : Over <-<-<-')


# 拟合图
def draw_true1pred_img(save_path, pred, label, series_len: int = 365):
    # 处理一下数据
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    # 绘图
    plt.figure(figsize=(12, 5))
    days = np.arange(1, series_len + 1)
    plt.plot(days, label, label='GroundTruth', marker='o')
    plt.plot(days, pred, label='Prediction', marker='x')
    plt.title('GroundTruth vs Prediction')
    plt.xlabel('time series')
    plt.ylabel('value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=600)
    plt.close()


# 对比柱状图
def draw_bar(save_path, values, labels, y_name, x_name=None, title=None):
    color_12 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
    ]

    # 2. 画图
    plt.figure(figsize=(6, 4))  # 图幅大小可调整
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    seq = max(values) - min(values)
    plt.ylim(min(values) - 0.3 * seq, max(values) + 0.3 * seq)
    bars = plt.bar(labels, values, color=color_12[:len(values)], width=0.35)

    # 3. 坐标轴与标题
    plt.ylabel(y_name)  # Y 轴名称
    if x_name is not None:
        plt.xlabel(x_name)  # X 轴名称（可选）
    if title is not None:
        plt.title(title)  # 图标题（可选）

    # 4. 数值标签（可选：在柱子上方标出具体数值）
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def draw_multiBar_in1x(save_path, multi_values, bar_type_names, labels,
                       y_name, x_name=None, title=None):
    """
    multi_values: 二维 list/array，shape=(组数, 类别数)
    bar_type_names: 每组图例名称，长度=组数
    labels: 类别名称，长度=类别数
    """
    color_12 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
    ]

    x = np.arange(len(labels))  # 类别坐标 0,1,2...
    n_group = len(multi_values)  # 组数
    width = 0.35  # 单根柱宽度
    # 关键改动：计算每组柱子的偏移量，使整组居中
    offsets = np.arange(n_group) - (n_group - 1) / 2  # [-1, 0, 1] 当 n_group=3

    plt.figure(figsize=(7, 4))
    bars_types = []
    for i, (values, name) in enumerate(zip(multi_values, bar_type_names)):
        # 关键改动：x + offsets[i]*width 实现并排
        bars = plt.bar(x + offsets[i] * width, values, width,
                       label=name, color=color_12[i % len(color_12)])
        bars_types.append(bars)

    plt.ylabel(y_name)
    if x_name is not None:
        plt.xlabel(x_name)
    if title is not None:
        plt.title(title)
    plt.xticks(x, labels)

    # Y 轴范围：按全局最大/最小值留边
    flat_vals = np.array(multi_values).ravel()
    seq = flat_vals.max() - flat_vals.min()
    plt.ylim(max(flat_vals.min() - 0.1 * seq, 0), flat_vals.max() + 0.3 * seq)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 数值标签
    for bars in bars_types:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# def draw_curve(x_values, y_values, x_name: str, y_name: str, title: str, save_path: str):
#     x_values = np.array(x_values)
#     y_values = np.array(y_values)
#     assert x_values.shape[0] == y_values.shape[0], "[code error] x_values shape should be equal with y_values shape!"
#
#     # # 创建图形
#     # plt.figure(figsize=(10, 6))
#     #
#     # # 绘制曲线图
#     # plt.plot(x_values, y_values,
#     #          marker='o',  # 标记点形状
#     #          linewidth=2,  # 线宽
#     #          markersize=8,  # 标记点大小
#     #          color='#1f77b4',  # 线条颜色
#     #          markerfacecolor='white',  # 标记点填充色
#     #          markeredgecolor='#1f77b4',  # 标记点边框色
#     #          markeredgewidth=2)  # 标记点边框宽度
#     #
#     # # 添加标题和标签
#     # plt.title(title)
#     # plt.xlabel(x_name, fontsize=12)
#     # plt.ylabel(y_name, fontsize=12)
#     #
#     # # 显示网格
#     # # plt.grid(True, alpha=0.3, linestyle='--')
#     #
#     # # 在点上标注数值
#     # for i, (x, y) in enumerate(zip(x_values, y_values)):
#     #     plt.annotate(f'{y:.3f}',
#     #                  (x, y),
#     #                  textcoords="offset points",
#     #                  xytext=(0, 10),
#     #                  ha='center',
#     #                  fontsize=9)
#     #
#     # # 调整布局
#     # plt.tight_layout()
#     #
#     # # 显示图形
#     # # plt.show()
#     #
#     # # 如果需要保存图片
#     # plt.savefig(save_path, dpi=600, bbox_inches='tight')
#
#     # ==================== 绘图 ====================
#     # special_x = 768
#
#     fig, ax = plt.subplots(figsize=(12, 5))
#
#     # 绘制蓝色曲线
#     ax.plot(x_values, y_values,
#             color='blue',  # 蓝色线条
#             linewidth=2,  # 线宽
#             label='Error curve')  # 图例标签
#
#     # 绘制普通数据点（圆形标记）
#     ax.plot(x_values, y_values,
#             marker='o',  # 圆形标记
#             color='black',  # 标记颜色
#             markersize=4,  # 标记大小
#             linestyle='none')  # 不画线，只标记点
#
#     # 在特殊x值位置添加红框黄星
#     # if special_x in x_values:
#     #     # 找到对应的y值
#     #     special_y = errors[embedding_sizes.index(special_x)]
#     #
#     #     # 绘制星星：红色边框，黄色填充
#     #     ax.plot(special_x, special_y,
#     #             marker='*',  # 星星标记
#     #             markersize=15,  # 星星大小
#     #             markeredgecolor='red',  # 边框红色
#     #             markerfacecolor='yellow',  # 填充黄色
#     #             markeredgewidth=2,  # 边框宽度
#     #             linestyle='none',  # 不连线
#     #             label=f'x={special_x}',  # 图例标签
#     #             zorder=5)  # 置于顶层
#
#     # ==================== 样式设置 ====================
#     # 设置标题和坐标轴标签
#     ax.set_title(title)
#     ax.set_xlabel(x_name)
#     ax.set_ylabel(y_name)
#
#     # 显示图例
#     ax.legend()
#
#     # 设置网格线（灰色，alpha=0.3）
#     plt.grid(alpha=0.3)
#
#     # 调整布局
#     plt.tight_layout()
#
#     # ==================== 显示/保存 ====================
#     # 显示图形
#     # plt.show()
#
#     # 保存高清图片（dpi=600）
#     plt.savefig(save_path, dpi=600)
#     plt.close()
#
#
# def draw_curve_and_special_start(x_values, y_values, spacial_x, x_name, y_name, title: str, save_path: str):
#     x_values = np.array(x_values)
#     y_values = np.array(y_values)
#     assert x_values.shape[0] == y_values.shape[0], "[code error] x_values shape should be equal with y_values shape!"
#
#     x_values = x_values.tolist()
#
#     fig, ax = plt.subplots(figsize=(12, 5))
#
#     # 绘制蓝色曲线
#     ax.plot(x_values, y_values,
#             color='blue',  # 蓝色线条
#             linewidth=2,  # 线宽
#             label='Error curve')  # 图例标签
#
#     # 绘制普通数据点（圆形标记）
#     ax.plot(x_values, y_values,
#             marker='o',  # 圆形标记
#             color='blue',  # 标记颜色
#             markersize=4,  # 标记大小
#             linestyle='none')  # 不画线，只标记点
#
#     # 在特殊x值位置添加红框黄星
#     if spacial_x in x_values:
#         # 找到对应的y值
#         special_y = y_values[x_values.index(spacial_x)]
#
#         # 绘制星星：红色边框，黄色填充
#         ax.plot(spacial_x, special_y,
#                 marker='*',  # 星星标记
#                 markersize=10,  # 星星大小
#                 markeredgecolor='red',  # 边框红色
#                 markerfacecolor='yellow',  # 填充黄色
#                 markeredgewidth=2,  # 边框宽度
#                 linestyle='none',  # 不连线
#                 label=f'x={spacial_x}',  # 图例标签
#                 zorder=5)  # 置于顶层
#
#     # ==================== 样式设置 ====================
#     # 设置标题和坐标轴标签
#     ax.set_title(title)
#     ax.set_xlabel(x_name)
#     ax.set_ylabel(y_name)
#
#     # 显示图例
#     ax.legend()
#
#     # 设置网格线（灰色，alpha=0.3）
#     # ax.grid(alpha=0.3, color='gray')
#     plt.grid(alpha=0.3)
#
#     # 调整布局
#     plt.tight_layout()
#
#     # ==================== 显示/保存 ====================
#     # 显示图形
#     # plt.show()
#
#     # 保存高清图片（dpi=600）
#     plt.savefig(save_path, dpi=600)
#     plt.close()


def draw_line(x_values, y_values, x_name: str, y_name: str, title: str, save_path: str, split_x=None, split_y=None, ):
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    assert x_values.shape[0] == y_values.shape[0], "[code error] x_values shape should be equal with y_values shape!"

    # # 创建图形
    # plt.figure(figsize=(10, 6))
    #
    # # 绘制曲线图
    # plt.plot(x_values, y_values,
    #          marker='o',  # 标记点形状
    #          linewidth=2,  # 线宽
    #          markersize=8,  # 标记点大小
    #          color='#1f77b4',  # 线条颜色
    #          markerfacecolor='white',  # 标记点填充色
    #          markeredgecolor='#1f77b4',  # 标记点边框色
    #          markeredgewidth=2)  # 标记点边框宽度
    #
    # # 添加标题和标签
    # plt.title(title)
    # plt.xlabel(x_name, fontsize=12)
    # plt.ylabel(y_name, fontsize=12)
    #
    # # 显示网格
    # # plt.grid(True, alpha=0.3, linestyle='--')
    #
    # # 在点上标注数值
    # for i, (x, y) in enumerate(zip(x_values, y_values)):
    #     plt.annotate(f'{y:.3f}',
    #                  (x, y),
    #                  textcoords="offset points",
    #                  xytext=(0, 10),
    #                  ha='center',
    #                  fontsize=9)
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 显示图形
    # # plt.show()
    #
    # # 如果需要保存图片
    # plt.savefig(save_path, dpi=600, bbox_inches='tight')

    # ==================== 绘图 ====================
    # special_x = 768

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制蓝色曲线
    ax.plot(x_values, y_values,
            color='blue',  # 蓝色线条
            linewidth=2,  # 线宽
            label='Error curve')  # 图例标签

    # 绘制普通数据点（圆形标记）
    ax.plot(x_values, y_values,
            marker='o',  # 圆形标记
            color='black',  # 标记颜色
            markersize=4,  # 标记大小
            linestyle='none',  # 不画线，只标记点
            label=f"{x_name} --- {y_name}")

    # 在散点上方标注数值
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.annotate(f'{y:.3f}',  # 显示3位小数
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),  # 在点上方8个像素处
                    ha='center',
                    fontsize=8,
                    color='black',
                    fontweight='bold')

    # 在特殊x值位置添加红框黄星
    # if special_x in x_values:
    #     # 找到对应的y值
    #     special_y = errors[embedding_sizes.index(special_x)]
    #
    #     # 绘制星星：红色边框，黄色填充
    #     ax.plot(special_x, special_y,
    #             marker='*',  # 星星标记
    #             markersize=15,  # 星星大小
    #             markeredgecolor='red',  # 边框红色
    #             markerfacecolor='yellow',  # 填充黄色
    #             markeredgewidth=2,  # 边框宽度
    #             linestyle='none',  # 不连线
    #             label=f'x={special_x}',  # 图例标签
    #             zorder=5)  # 置于顶层

    # ==================== 样式设置 ====================
    # 设置标题和坐标轴标签
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    # 显示图例
    ax.legend()

    # 设置网格线（灰色，alpha=0.3）
    if split_x:
        ax.xaxis.set_major_locator(plt.MultipleLocator(split_x))
    if split_y:
        ax.yaxis.set_major_locator(plt.MultipleLocator(split_y))
    plt.grid(alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # ==================== 显示/保存 ====================
    # 显示图形
    # plt.show()

    # 保存高清图片（dpi=600）
    plt.savefig(save_path, dpi=600)
    plt.close()


def draw_line_and_special_star(x_values, y_values, spacial_x, x_name, y_name, title: str, save_path: str, split_x=None,
                               split_y=None, ):
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    assert x_values.shape[0] == y_values.shape[0], "[code error] x_values shape should be equal with y_values shape!"

    x_values = x_values.tolist()

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制蓝色曲线
    ax.plot(x_values, y_values,
            color='blue',  # 蓝色线条
            linewidth=2,  # 线宽
            label='Error curve')  # 图例标签

    # 绘制普通数据点（圆形标记）
    ax.plot(x_values, y_values,
            marker='o',  # 圆形标记
            color='black',  # 标记颜色
            markersize=4,  # 标记大小
            linestyle='none',  # 不画线，只标记点
            label=f"{x_name} --- {y_name}")

    # 在散点上方标注数值
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.annotate(f'{y:.3f}',  # 显示3位小数
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),  # 在点上方8个像素处
                    ha='center',
                    fontsize=8,
                    color='black',
                    fontweight='bold')

    # 在特殊x值位置添加红框黄星
    if spacial_x in x_values:
        # 找到对应的y值
        special_y = y_values[x_values.index(spacial_x)]

        # 绘制星星：红色边框，黄色填充
        ax.plot(spacial_x, special_y,
                marker='*',  # 星星标记
                markersize=10,  # 星星大小
                markeredgecolor='red',  # 边框红色
                markerfacecolor='yellow',  # 填充黄色
                markeredgewidth=2,  # 边框宽度
                linestyle='none',  # 不连线
                label=f'{x_name} = {spacial_x}',  # 图例标签
                zorder=5)  # 置于顶层

    # ==================== 样式设置 ====================
    # 设置标题和坐标轴标签
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    # 显示图例
    ax.legend()

    # 设置网格线（灰色，alpha=0.3）
    # ax.grid(alpha=0.3, color='gray')
    if split_x:
        ax.xaxis.set_major_locator(plt.MultipleLocator(split_x))
    if split_y:
        ax.yaxis.set_major_locator(plt.MultipleLocator(split_y))
    plt.grid(alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # ==================== 显示/保存 ====================
    # 显示图形
    # plt.show()

    # 保存高清图片（dpi=600）
    plt.savefig(save_path, dpi=600)
    plt.close()


def draw_curve(x_values, y_values, x_name: str, y_name: str, title: str, save_path: str, split_x=None, split_y=None, ):
    """绘制平滑曲线图"""
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    assert x_values.shape[0] == y_values.shape[0], "[code error] x_values shape should be equal with y_values shape!"

    # 创建平滑曲线数据点
    x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
    spl = make_interp_spline(x_values, y_values, k=3)
    y_smooth = spl(x_smooth)

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制平滑曲线（蓝色）
    ax.plot(x_smooth, y_smooth,
            color='blue',
            linewidth=2,
            label='Error curve')

    # 绘制原始数据点（圆形标记，黑色）
    ax.plot(x_values, y_values,
            marker='o',
            color='black',
            markersize=4,
            linestyle='none',
            label=f"{x_name} --- {y_name}")

    # 在散点上方标注数值
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.annotate(f'{y:.3f}',  # 显示3位小数
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),  # 在点上方8个像素处
                    ha='center',
                    fontsize=8,
                    color='black',
                    fontweight='bold')

    # 设置标题和坐标轴标签
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    # 显示图例
    ax.legend()

    # 设置网格线
    if split_x:
        ax.xaxis.set_major_locator(plt.MultipleLocator(split_x))
    if split_y:
        ax.yaxis.set_major_locator(plt.MultipleLocator(split_y))
    plt.grid(alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=600)
    plt.close()


def draw_curve_and_special_star(x_values, y_values, special_x, x_name, y_name, title: str, save_path: str, split_x=None,
                                split_y=None, ):
    """绘制平滑曲线图，并在曲线上的特殊位置标记星号"""
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    assert x_values.shape[0] == y_values.shape[0], "[code error] x_values shape should be equal with y_values shape!"

    # 创建平滑曲线数据点
    x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
    spl = make_interp_spline(x_values, y_values, k=3)
    y_smooth = spl(x_smooth)

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制平滑曲线（蓝色）
    ax.plot(x_smooth, y_smooth,
            color='blue',
            linewidth=2,
            label='Error curve')

    # 绘制原始数据点（圆形标记）
    ax.plot(x_values, y_values,
            marker='o',
            color='black',
            markersize=4,
            linestyle='none',
            label=f"{x_name} --- {y_name}")

    # 在散点上方标注数值
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.annotate(f'{y:.3f}',  # 显示3位小数
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),  # 在点上方8个像素处
                    ha='center',
                    fontsize=8,
                    color='black',
                    fontweight='bold')

    # 在特殊x值位置添加红框黄星（在曲线上）
    if x_values.min() <= special_x <= x_values.max():
        # 计算特殊点在平滑曲线上的y值
        x_values = x_values.tolist()
        special_y = y_values[x_values.index(special_x)]

        # 绘制星星：红色边框，黄色填充
        ax.plot(special_x, special_y,
                marker='*',
                markersize=12,
                markeredgecolor='red',
                markerfacecolor='yellow',
                markeredgewidth=2,
                linestyle='none',
                label=f'{x_name} = {special_x} (on curve)',
                zorder=5)

    # 设置标题和坐标轴标签
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    # 显示图例
    ax.legend()

    # 设置网格线
    if split_x:
        ax.xaxis.set_major_locator(plt.MultipleLocator(split_x))
    if split_y:
        ax.yaxis.set_major_locator(plt.MultipleLocator(split_y))
    plt.grid(alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=600)
    plt.close()


def draw_curve_and_special_star_smooth(x_values, y_values, special_x, x_name, y_name, title: str, save_path: str,
                                       split_x=None, split_y=None, ):
    """绘制平滑曲线图，并在曲线上的特殊位置标记星号"""
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    assert x_values.shape[0] == y_values.shape[0], "[code error] x_values shape should be equal with y_values shape!"

    # 创建平滑曲线数据点
    x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
    spl = make_interp_spline(x_values, y_values, k=3)
    y_smooth = spl(x_smooth)

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制平滑曲线（蓝色）
    ax.plot(x_smooth, y_smooth,
            color='blue',
            linewidth=2,
            label='Error curve')

    # 绘制原始数据点（圆形标记）
    ax.plot(x_values, y_values,
            marker='o',
            color='black',
            markersize=4,
            linestyle='none',
            label=f"{x_name} --- {y_name}")

    # 在散点上方标注数值
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.annotate(f'{y:.3f}',  # 显示3位小数
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),  # 在点上方8个像素处
                    ha='center',
                    fontsize=8,
                    color='black',
                    fontweight='bold')

    # 在特殊x值位置添加红框黄星（在曲线上）
    if x_values.min() <= special_x <= x_values.max():
        # 计算特殊点在平滑曲线上的y值
        special_y = spl(special_x)

        # 绘制星星：红色边框，黄色填充
        ax.plot(special_x, special_y,
                marker='*',
                markersize=12,
                markeredgecolor='red',
                markerfacecolor='yellow',
                markeredgewidth=2,
                linestyle='none',
                label=f'{x_name} = {special_x} (on curve)',
                zorder=5)

    # 设置标题和坐标轴标签
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    # 显示图例
    ax.legend()

    # 设置网格线
    if split_x:
        ax.xaxis.set_major_locator(plt.MultipleLocator(split_x))
    if split_y:
        ax.yaxis.set_major_locator(plt.MultipleLocator(split_y))
    plt.grid(alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=600)
    plt.close()
