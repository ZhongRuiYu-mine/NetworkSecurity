import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score
)
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

# 中文字体（如无中文需求可删）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


# ==================== 1. 数据加载与预处理 ====================
def load_and_clean(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


monday = load_and_clean('Monday-WorkingHours.pcap_ISCX.csv')
tuesday = load_and_clean('Tuesday-WorkingHours.pcap_ISCX.csv')

feature_cols = [c for c in monday.columns if c not in ['Label', 'Timestamp']]
feature_cols = [c for c in feature_cols if c in tuesday.columns]

# 测试集标签
y_test = (tuesday['Label'] != 'BENIGN').astype(int).values
print(f"[数据] Monday 正常样本: {len(monday)}, Tuesday 总样本: {len(tuesday)}, "
      f"其中攻击: {y_test.sum()} ({y_test.mean()*100:.2f}%)")


# ==================== 2. 分布漂移检查（Monday正常 vs Tuesday正常） ====================
# 只看两边都是 BENIGN 的部分，检查特征分布是否一致
tuesday_normal = tuesday[tuesday['Label'] == 'BENIGN']

print("\n[分布漂移] 对每个特征做 KS 检验（Monday正常 vs Tuesday正常），"
      "只显示 p<0.01 的显著漂移特征：")
drift_count = 0
for c in feature_cols:
    stat, p = ks_2samp(monday[c].values, tuesday_normal[c].values)
    if p < 0.01:
        drift_count += 1
        if drift_count <= 8:  # 只打印前几个，避免刷屏
            print(f"  {c:40s} KS={stat:.3f}, p={p:.2e}")
print(f"  ... 共 {drift_count}/{len(feature_cols)} 个特征存在显著漂移")


# ==================== 3. 构造两种预处理版本，用于消融 ====================
X_train_raw = monday[feature_cols].values
X_test_raw = tuesday[feature_cols].values

scaler = StandardScaler().fit(X_train_raw)
X_train_scaled = scaler.transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

variants = {
    "raw": (X_train_raw, X_test_raw),
    "scaled": (X_train_scaled, X_test_scaled),
}


# ==================== 4. 工具函数 ====================
def evaluate_scores(y_true, scores, name="model"):
    """
    scores: 越高越异常
    返回：dict of metrics + 曲线数据
    """
    # AUC 方向校验
    auc = roc_auc_score(y_true, scores)
    flipped = False
    if auc < 0.5:
        scores = -scores
        auc = roc_auc_score(y_true, scores)
        flipped = True
        print(f"  [警告] {name}: 分数方向相反，已自动翻转")

    ap = average_precision_score(y_true, scores)

    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, pr_thresh = precision_recall_curve(y_true, scores)

    # 阈值扫描：找 F1 最大的阈值
    thresholds = np.unique(scores)
    best_f1, best_thr = 0, None
    for t in thresholds:
        yp = (scores >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    # 先验异常率对应的阈值（测试集真实异常率作为 oracle 上界）
    prior_thr = np.quantile(scores, 1 - y_true.mean())
    yp_prior = (scores >= prior_thr).astype(int)

    return {
        "name": name,
        "auc": auc,
        "ap": ap,
        "flipped": flipped,
        "fpr": fpr, "tpr": tpr,
        "prec": prec, "rec": rec,
        "best_f1": best_f1, "best_thr": best_thr,
        "prior_thr": prior_thr,
        "y_pred_prior": yp_prior,
        "y_pred_bestf1": (scores >= best_thr).astype(int),
        "scores": scores,
    }


# ==================== 5. 训练 + 评估（消融） ====================
results = {}
models = {}

for variant_name, (Xtr, Xte) in variants.items():
    print(f"\n{'='*70}")
    print(f"变体: {variant_name}")
    print('='*70)

    iso = IsolationForest(
        n_estimators=200,
        max_samples=256,          # 标准做法：固定采样量，避免大样本下树过深
        contamination='auto',     # 只影响 predict 阈值，我们不使用 predict
        random_state=42,
        n_jobs=-1
    )
    iso.fit(Xtr)

    # score_samples: 越高越正常 → 取负
    scores = -iso.score_samples(Xte)

    res = evaluate_scores(y_test, scores, name=variant_name)
    results[variant_name] = res
    models[variant_name] = iso

    print(f"  AUC-ROC : {res['auc']:.4f}")
    print(f"  PR-AUC  : {res['ap']:.4f}  ← 异常检测更关注这个")
    print(f"  Best F1 : {res['best_f1']:.4f} @ thr={res['best_thr']:.4f}")
    print(f"  先验异常率={y_test.mean():.4f} 时:")
    print(classification_report(y_test, res['y_pred_prior'],
                                target_names=['Normal', 'Attack'], digits=4))


# ==================== 6. 可视化 ====================
best_name = max(results, key=lambda k: results[k]['ap'])
best = results[best_name]
print(f"\n[可视化] 使用变体 '{best_name}' (PR-AUC 最高)")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# --- (1) 异常分数分布 ---
ax = fig.add_subplot(gs[0, 0])
scores = best['scores']
ax.hist(scores[y_test == 0], bins=60, alpha=0.6, label='Normal', color='steelblue', density=True)
ax.hist(scores[y_test == 1], bins=60, alpha=0.6, label='Attack', color='crimson', density=True)
ax.axvline(best['best_thr'], color='black', linestyle='--', label=f"BestF1 thr")
ax.axvline(best['prior_thr'], color='green', linestyle=':', label=f"Prior thr")
ax.set_title(f'异常分数分布 ({best_name})')
ax.set_xlabel('Anomaly Score (越高越异常)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)

# --- (2) ROC 曲线 ---
ax = fig.add_subplot(gs[0, 1])
for name, r in results.items():
    ax.plot(r['fpr'], r['tpr'], label=f"{name} (AUC={r['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.set_title('ROC 曲线')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.legend(fontsize=8)

# --- (3) PR 曲线 ---
ax = fig.add_subplot(gs[0, 2])
for name, r in results.items():
    ax.plot(r['rec'], r['prec'], label=f"{name} (AP={r['ap']:.4f})")
ax.axhline(y_test.mean(), color='gray', linestyle='--', lw=0.8,
           label=f"baseline (prior={y_test.mean():.3f})")
ax.set_title('Precision-Recall 曲线')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(fontsize=8)

# --- (4) 阈值扫描：F1 / Precision / Recall ---
ax = fig.add_subplot(gs[1, 0])
scores = best['scores']
thresholds = np.quantile(scores, np.linspace(0.5, 0.999, 100))
prec_list, rec_list, f1_list = [], [], []
for t in thresholds:
    yp = (scores >= t).astype(int)
    tp = ((yp == 1) & (y_test == 1)).sum()
    fp = ((yp == 1) & (y_test == 0)).sum()
    fn = ((yp == 0) & (y_test == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f = 2 * p * r / (p + r) if (p + r) else 0
    prec_list.append(p); rec_list.append(r); f1_list.append(f)
ax.plot(thresholds, prec_list, label='Precision')
ax.plot(thresholds, rec_list, label='Recall')
ax.plot(thresholds, f1_list, label='F1', lw=2)
ax.axvline(best['best_thr'], color='k', linestyle='--', alpha=0.6)
ax.set_title('阈值扫描 (P/R/F1)')
ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
ax.legend(fontsize=8)

# --- (5) 混淆矩阵 @ Best F1 ---
ax = fig.add_subplot(gs[1, 1])
cm = confusion_matrix(y_test, best['y_pred_bestf1'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
ax.set_title(f"混淆矩阵 @ BestF1 thr ({best_name})")
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

# --- (6) 混淆矩阵 @ Prior 异常率 ---
ax = fig.add_subplot(gs[1, 2])
cm2 = confusion_matrix(y_test, best['y_pred_prior'])
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
ax.set_title(f"混淆矩阵 @ 先验异常率 ({best_name})")
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

# --- (7) 分位数阈值分析（保留你原来的思路，但画图） ---
ax = fig.add_subplot(gs[2, 0])
alphas = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]
f1s = []
for a in alphas:
    thr = np.quantile(scores, 1 - a)
    yp = (scores >= thr).astype(int)
    f1s.append(f1_score(y_test, yp, zero_division=0))
ax.bar([str(a) for a in alphas], f1s, color='teal', alpha=0.7)
ax.set_title('分位数阈值下的 F1')
ax.set_xlabel('alpha (异常比例)'); ax.set_ylabel('F1')

# --- (8) 特征重要性（近似）：用 iTree 的 path length 贡献不易算，
#      改为展示各特征的异常分数相关性（用单特征 score 的 abs Spearman）
ax = fig.add_subplot(gs[2, 1:])
# 用单特征训练 IF 太慢，改为计算每个特征与异常分数的 |Spearman 相关|
from scipy.stats import spearmanr
Xte = variants[best_name][1]
corrs = []
for i, c in enumerate(feature_cols):
    rho, _ = spearmanr(Xte[:, i], scores)
    corrs.append(abs(rho))
corrs = np.array(corrs)
top_idx = np.argsort(corrs)[-15:][::-1]
ax.barh([feature_cols[i] for i in top_idx][::-1],
        corrs[top_idx][::-1], color='slateblue', alpha=0.8)
ax.set_title(f'与异常分数 |Spearman| 最高的 Top15 特征 ({best_name})')
ax.set_xlabel('|Spearman correlation|')

plt.suptitle(f'Isolation Forest 异常检测 — CICIDS2017 (Monday→Tuesday)',
             fontsize=14, y=0.995)
plt.savefig('if_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[可视化] 已保存到 if_analysis.png")


# ==================== 7. 最终汇总表 ====================
print("\n" + "="*70)
print("消融汇总")
print("="*70)
print(f"{'variant':<10} {'AUC':>8} {'PR-AUC':>8} {'BestF1':>8}")
for name, r in results.items():
    print(f"{name:<10} {r['auc']:>8.4f} {r['ap']:>8.4f} {r['best_f1']:>8.4f}")