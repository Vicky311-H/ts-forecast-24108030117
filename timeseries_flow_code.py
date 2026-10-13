# -*- coding: utf-8 -*-
"""
时间序列预测与可视化实战（前沿版）—— 基于交通流数据 flow.csv
=================================================================
配套教学文件: Day5_TimeSeries_Forecast_DeepTheory.html

    第1步  认识数据     —— 事件级数据聚合成规则时间序列
    第2步  平稳性       —— ADF；理解"单位根 ≠ 季节性"这个关键区别
    第3步  自相关       —— ACF / PACF，用 lag-24 峰读出日周期
    第4步  分解         —— 趋势 / 季节 / 残差
    第5步  评估协议     —— 滚动起源交叉验证 + MASE（现代标准）
    第6步  必备基线     —— Naive / Seasonal Naive（必须打败的检察官）
    第7步  现代线性族   —— Ridge-AR / NLinear / DLinear / 季节差分线性
    第8步  残差诊断     —— 残差 ACF + Ljung-Box 白噪声检验
    第9步  权重视图     —— 打开"黑箱"，看线性模型到底学了什么
    第10步 可视化       —— 9 张图落盘 + results.json

【为什么课堂上用线性模型，而不是 Transformer？】
2023 年起一系列实证工作反复发现：在中小规模、周期性强 的单变量序列上，
**带实例归一化/趋势-季节分解的线性模型往往打得过复杂的 Transformer 族**
（Autoformer 自身消融、UrbanAI 2025 挑战赛 arXiv:2512.10866）。
GIFT-Eval 榜单（arXiv:2410.10393，Salesforce）显示 PatchTST 与最新基础模型
（Chronos-2 / Moirai 2.0 / TimesFM 3.0）差距有限，而 Seasonal Naive 仍是
不可绕过的参照系。本脚本用**闭式岭回归**实现这一族：无需 GPU、秒级出结果、
结果确定可复现，适合把"方法选择的判断依据"讲透。

运行：
    python timeseries_flow_code.py
    python timeseries_flow_code.py --alphas 0.1 1 10 100

环境：
    pip install pandas numpy matplotlib scikit-learn statsmodels
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

CLIP_LOW, CLIP_HIGH = 10.0, 80.0
LOOK_BACK, HORIZON, N_FOLDS = 24, 24, 3
PERIOD = 24          # 季节周期（由 ACF 的 lag-24 峰读出）


def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {name}")
    return name


# ============================================================================
# 指标
# ============================================================================
def naive_scale(train):
    """MASE 分母：训练段朴素预测误差均值。只能用训练段，避免泄露。"""
    return float(np.mean(np.abs(np.diff(np.asarray(train, dtype=float)))))


def metrics(y_true, y_pred, scale):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MASE": float(np.mean(np.abs(y_true - y_pred)) / scale)}


# ============================================================================
# 第1步：认识数据
# ============================================================================
def build_series(csv_path):
    print("\n=== 第1步：认识数据 ===")
    df = pd.read_csv(csv_path)
    n_raw = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    sp = df["Speed (mph)"].astype(float)
    mask = sp.between(CLIP_LOW, CLIP_HIGH)
    n_bad = int((~mask).sum())
    df = df.loc[mask]
    df["day"], df["hour"] = df["Date"].dt.date, df["Date"].dt.hour
    hourly = df.groupby(["day", "hour"])["Speed (mph)"].mean().reset_index()
    days = sorted(hourly["day"].unique())
    print(f"  原始事件 {n_raw:,} | 剔除异常速度 {n_bad:,} | 保留 {len(df):,}")
    print(f"  有效 {len(days)} 天 x 24 小时 -> 规则序列 {len(days) * 24} 点")
    hourly = hourly.sort_values(["day", "hour"])
    idx = pd.date_range(start="2018-02-02", periods=len(hourly), freq="h")
    ts = pd.Series(hourly["Speed (mph)"].values.astype(float), index=idx, name="speed_mph")
    typical = df.groupby("hour")["Speed (mph)"].mean().reindex(range(24)).values
    return ts, typical, n_raw, n_bad


# ============================================================================
# 第2步：平稳性
# ============================================================================
def stationarity(ts):
    print("\n=== 第2步：平稳性检验 (ADF) ===")

    def adf(x, label):
        res = adfuller(x, autolag="AIC", result_object=False)
        p = res[1]
        print(f"  {label}: 统计量={res[0]:.3f}, p={p:.3e} -> {'平稳' if p < 0.05 else '非平稳'}")
        return {"adf": float(res[0]), "p": float(p), "stationary": bool(p < 0.05)}

    return adf(ts.values, "原始序列"), adf(ts.diff().dropna().values, "一阶差分后")


# ============================================================================
# 第5步：滚动起源交叉验证
# ============================================================================
def rolling_origin_folds(n, look_back=LOOK_BACK, horizon=HORIZON, n_folds=N_FOLDS):
    folds, min_train = [], look_back + horizon + 24
    for k in range(n_folds + 1):
        test_end = min_train + horizon * (k + 1)
        test_start = test_end - horizon
        if test_end > n:
            break
        folds.append({"train": (0, test_start), "test": (test_start, test_end),
                      "name": f"折{k + 1}" if k < n_folds else "末端保留"})
    return folds


# ============================================================================
# 闭式岭回归工具
# ============================================================================
def make_xy(y, look_back, horizon):
    X, Y = [], []
    for i in range(look_back, len(y) - horizon + 1):
        X.append(y[i - look_back:i])
        Y.append(y[i:i + horizon])
    return np.array(X), np.array(Y)


def ridge_fit(X, Y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)


def ridge_pred(W, X):
    return X @ W


def trailing_ma(x, k):
    """因果移动平均：只用到当前位置为止的 k 个点。
    居中窗口会看到未来 -> 直接数据泄露，课堂上要反复强调。"""
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = np.mean(x[max(0, i - k + 1):i + 1])
    return out


def select_alpha(Xtr, Ytr, alphas, horizon):
    """训练段尾部做内层验证选正则强度（时间有序，绝不打乱）。"""
    n_val = max(horizon, len(Xtr) // 5)
    n_fit = len(Xtr) - n_val
    if n_fit < 10:
        return alphas[0]
    best, best_a = None, alphas[0]
    for a in alphas:
        W = ridge_fit(Xtr[:n_fit], Ytr[:n_fit], a)
        e = np.mean(np.abs(ridge_pred(W, Xtr[n_fit:]) - Ytr[n_fit:]))
        if best is None or e < best:
            best, best_a = e, a
    return best_a


class ModernLinear:
    """现代线性模型族。核心差异全部集中在 _feat（输入表征）与 _target（目标空间）。"""

    def __init__(self, kind, look_back=LOOK_BACK, horizon=HORIZON,
                 alphas=(0.1, 1.0, 10.0, 100.0), ma_kernel=25):
        self.kind, self.L, self.H = kind, look_back, horizon
        self.alphas, self.ma_k = list(alphas), ma_kernel
        self.w_ = None
        self.fmu_ = self.fsd_ = self.tmu_ = self.tsd_ = None

    # ---- 输入表征 ----
    def _feat(self, W):
        if self.kind == "ridge_ar":
            return W                                    # 原始窗口直接回归
        if self.kind == "nlinear":
            return W - W[:, -1:]                        # 实例归一化（减末值）
        if self.kind == "dlinear":
            tr = np.array([trailing_ma(r, self.ma_k) for r in W])
            return np.hstack([tr, W - tr])              # 因果分解：趋势头 + 季节头
        raise ValueError(self.kind)

    # ---- 目标空间（与表征必须一致，否则量纲错位）----
    def _target(self, X, Y):
        if self.kind == "nlinear":
            return Y - X[:, -1:]                       # 同样在"相对末值"空间
        return Y

    def fit(self, y):
        X, Y = make_xy(y, self.L, self.H)
        F, T = self._feat(X), self._target(X, Y)
        self.fmu_, self.fsd_ = F.mean(0), F.std(0) + 1e-8
        self.tmu_, self.tsd_ = T.mean(), T.std() + 1e-8
        Fn, Tn = (F - self.fmu_) / self.fsd_, (T - self.tmu_) / self.tsd_
        self.alpha_ = select_alpha(Fn, Tn, self.alphas, self.H)
        self.w_ = ridge_fit(Fn, Tn, self.alpha_)
        return self

    def predict(self, y):
        w = y[-self.L:].reshape(1, -1)
        P = ridge_pred(self.w_, (self._feat(w) - self.fmu_) / self.fsd_)
        P = P * self.tsd_ + self.tmu_
        if self.kind == "nlinear":
            P = P + w[:, -1:]                           # 反归一化：加回基准值
        return P.flatten()

    def contributions(self):
        """各输入位置对未来 H 步的平均 |权重| 贡献（可解释性）。"""
        if self.kind == "dlinear":
            c = np.abs(self.w_[: self.L]) + np.abs(self.w_[self.L:])
        else:
            c = np.abs(self.w_)
        return c.mean(axis=1)


def fit_predict_seasdiff(y, horizon=HORIZON, period=PERIOD, alphas=(0.1, 1.0, 10.0, 100.0)):
    """季节差分线性模型：先在 z_t = y_t - y_{t-period} 的平稳空间做线性预测，
    再用最后 period 个真实值重建回原空间。
    这是"用线性模型实现 ARIMA 思想"的最小可复现版本（不递归、不漂移）。"""
    z = y[period:] - y[:-period]
    X, Y = make_xy(z, LOOK_BACK, horizon)
    fmu, fsd = X.mean(0), X.std(0) + 1e-8
    tmu, tsd = Y.mean(), Y.std() + 1e-8
    Fn, Tn = (X - fmu) / fsd, (Y - tmu) / tsd
    alpha = select_alpha(Fn, Tn, list(alphas), horizon)
    W = ridge_fit(Fn, Tn, alpha)
    zf = ridge_pred(W, (z[-LOOK_BACK:].reshape(1, -1) - fmu) / fsd) * tsd + tmu
    zf = zf.flatten()
    # 重建：ŷ[t+h] = y[t+h-period] + ẑ[t+h]，基准全部来自真实观测
    out = np.empty(horizon)
    for h in range(horizon):
        out[h] = y[len(y) - period + h] + zf[h]
    return out, {"alpha": alpha, "contrib": np.abs(W).mean(axis=1).tolist()}


# ============================================================================
# 单折：基线 + 现代线性族
# ============================================================================
def run_models(train, test, alphas):
    y, yte = np.asarray(train, float), np.asarray(test, float)
    scale, H = naive_scale(y), len(yte)
    out, detail = {}, {}

    out["Naive(随机游走)"] = np.repeat(y[-1], H)
    out["Seasonal Naive"] = np.resize(y[-PERIOD:], H)
    for kind, label in [("ridge_ar", "Ridge-AR(线性基线)"),
                        ("nlinear", "NLinear(实例归一化)"),
                        ("dlinear", "DLinear(趋势-季节双头)")]:
        m = ModernLinear(kind, alphas=alphas).fit(y)
        out[label] = m.predict(y)
        detail[label] = m
    sd_pred, sd_info = fit_predict_seasdiff(y, H, PERIOD, alphas)
    out["SeasonalDiff(季节差分线性)"] = sd_pred
    return out, detail, scale, sd_info


# ============================================================================
# 主流程
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "flow.csv"))
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    args = ap.parse_args()

    print("+" * 68)
    print(" 时间序列预测与可视化实战（前沿版） —  flow.csv 交通流车速")
    print(" 模型: 必备基线 + 现代线性族(闭式岭回归, 确定性可复现, 无需GPU)")
    print("+" * 68)

    ts, typical, n_raw, n_bad = build_series(args.data)
    adf_raw, adf_diff = stationarity(ts)
    diff = ts.diff().dropna()

    # ---------- 滚动起源交叉验证 ----------
    print("\n=== 第5步：滚动起源交叉验证 ===")
    folds = rolling_origin_folds(len(ts))
    cv_rows, cv_resid = {}, {}
    for f in folds:
        tr, te = ts.iloc[f["train"][0]:f["train"][1]], ts.iloc[f["test"][0]:f["test"][1]]
        preds, _, scale, _ = run_models(tr, te, args.alphas)
        for name, p in preds.items():
            cv_rows.setdefault(name, []).append(metrics(te.values, p, scale))
            cv_resid.setdefault(name, []).append(te.values - p)
        print(f"  {f['name']}: 训练 {len(tr)} -> 测试 {len(te)}")
    print(f"\n  {'模型':<30}{'CV MAE(均值±标准差)':>24}{'CV MASE':>12}")
    cv_summary = {}
    for name, rows in cv_rows.items():
        if len(rows) < N_FOLDS:
            continue
        mae = np.array([r["MAE"] for r in rows])
        mase = np.array([r["MASE"] for r in rows])
        cv_summary[name] = {"MAE_mean": float(mae.mean()), "MAE_std": float(mae.std()),
                            "MASE_mean": float(mase.mean()), "MASE_std": float(mase.std())}
        print(f"  {name:<30}{mae.mean():>13.3f} ± {mae.std():<8.3f}{mase.mean():>12.3f}")

    # ---------- 末端保留折 ----------
    final = [f for f in folds if f["name"] == "末端保留"][-1]
    train, test = ts.iloc[:final["train"][1]], ts.iloc[final["test"][0]:final["test"][1]]
    print(f"\n=== 第6-7步：末端保留折（训练 {len(train)} -> 测试 {len(test)}）===")
    preds, detail, scale, sd_info = run_models(train, test, args.alphas)
    table = {k: metrics(test.values, p, scale) for k, p in preds.items()}
    order = sorted(table, key=lambda k: table[k]["MASE"])
    print(f"  {'模型':<30}{'MAE':>10}{'RMSE':>10}{'MASE':>10}")
    for k in order:
        m = table[k]
        print(f"  {k:<30}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}{m['MASE']:>10.3f}")
    print("\n  MASE<1 = 优于朴素水平。Seasonal Naive 是 GIFT-Eval 等榜单的标准参照系。")

    # ---------- 残差诊断（用 CV 汇集的 72 个样本外残差，统计上更稳）----------
    print("\n=== 第8步：残差诊断 (Ljung-Box, 基于 CV 汇集残差) ===")
    resid_report = {}
    for name in order:
        pool = np.concatenate(cv_resid[name])
        lag = max(1, min(12, len(pool) // 5))
        lb = acorr_ljungbox(pool, lags=[lag], return_df=True)
        stat, pval = float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
        resid_report[name] = {"ljungbox_lag": lag, "stat": stat, "p": pval,
                              "pooled_n": int(len(pool))}
        print(f"  {name:<30} LB({lag})={stat:>7.2f}  p={pval:.3f}  -> "
              f"{'仍有结构(未榨干)' if pval < 0.05 else '近似白噪声(OK)'}")

    # ---------- 权重 ----------
    print("\n=== 第9步：线性模型权重（可解释性）===")
    weights = {}
    for label, m in detail.items():
        weights[label] = m.contributions().tolist()
        print(f"  {label:<30} 特征维度={m.w_.shape[0]:>3}  内层验证 alpha={m.alpha_}")
    weights["SeasonalDiff(季节差分线性)"] = sd_info["contrib"]
    print(f"  {'SeasonalDiff(季节差分线性)':<30} 特征维度={LOOK_BACK:>3}  "
          f"内层验证 alpha={sd_info['alpha']}")

    # ---------- 可视化 ----------
    print("\n=== 第10步：可视化落盘 ===")
    decomp = sm.tsa.seasonal_decompose(ts, period=PERIOD, model="additive")
    figs = {}

    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].plot(ts.index, ts.values, color="#2563eb", lw=1.0)
    ax[0].set_title("完整序列：8 天 × 24 小时 = 192 个逐时车速点", fontsize=12)
    ax[0].set_ylabel("车速 (mph)"); ax[0].grid(alpha=0.3)
    ax[1].plot(range(24), typical, "o-", color="#7c3aed", lw=1.6)
    ax[1].set_title("典型日曲线：两峰两谷的日周期", fontsize=12)
    ax[1].set_xlabel("小时 (0-23)"); ax[1].set_ylabel("平均车速 (mph)")
    ax[1].axvspan(6.5, 9.5, color="#fee2e2", alpha=0.5, label="早高峰")
    ax[1].axvspan(15.5, 18.5, color="#fef3c7", alpha=0.6, label="晚高峰")
    ax[1].legend(); ax[1].grid(alpha=0.3); fig.tight_layout()
    figs["eda"] = savefig(fig, "ts_01_eda.png")

    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    ax[0].plot(ts.values, color="#2563eb")
    ax[0].set_title(f"原始序列 (ADF p={adf_raw['p']:.2e}, 判为平稳)", fontsize=11); ax[0].grid(alpha=0.3)
    ax[1].plot(diff.values, color="#059669")
    ax[1].set_title(f"一阶差分 (ADF p={adf_diff['p']:.2e}, 判为平稳)", fontsize=11); ax[1].grid(alpha=0.3)
    roll = ts.rolling(PERIOD).mean()
    ax[2].plot(ts.values, color="#94a3b8", lw=0.8, label="原始")
    ax[2].plot(roll.values, color="#d97706", lw=2.0, label="24h 滚动均值")
    ax[2].set_title("滚动均值随昼夜大幅漂移：均值层的周期结构（ADF 测不到这一层）", fontsize=11)
    ax[2].legend(); ax[2].grid(alpha=0.3); fig.tight_layout()
    figs["stationarity"] = savefig(fig, "ts_02_stationarity.png")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    plot_acf(ts, lags=48, ax=ax[0, 0], color="#2563eb"); ax[0, 0].set_title("原始序列 ACF（lag24 峰 = 日周期）")
    plot_pacf(ts, lags=48, ax=ax[0, 1], color="#7c3aed"); ax[0, 1].set_title("原始序列 PACF")
    plot_acf(diff, lags=48, ax=ax[1, 0], color="#059669"); ax[1, 0].set_title("差分后 ACF")
    plot_pacf(diff, lags=48, ax=ax[1, 1], color="#d97706"); ax[1, 1].set_title("差分后 PACF（lag1 截尾）")
    fig.tight_layout(); figs["acf_pacf"] = savefig(fig, "ts_03_acf_pacf.png")

    fig, ax = plt.subplots(4, 1, figsize=(11, 9))
    for i, (s, c, t) in enumerate([(decomp.observed, "#2563eb", "观测值"),
                                   (decomp.trend, "#7c3aed", "趋势项"),
                                   (decomp.seasonal, "#059669", "季节项 (周期24)"),
                                   (decomp.resid, "#dc2626", "残差项")]):
        ax[i].plot(s, color=c); ax[i].set_ylabel(t.split()[0]); ax[i].set_title(t, fontsize=11)
        ax[i].grid(alpha=0.3)
    fig.tight_layout(); figs["decomp"] = savefig(fig, "ts_04_decomposition.png")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, f in enumerate(folds):
        tr0, tr1 = f["train"]; te0, te1 = f["test"]
        ax.barh(i, tr1 - tr0, left=tr0, color="#059669", alpha=0.75, height=0.55)
        ax.barh(i, te1 - te0, left=te0, color="#dc2626", alpha=0.85, height=0.55)
        ax.text((tr1 + te1) / 2, i, f["name"], va="center", ha="center",
                color="white", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(folds))); ax.set_yticklabels([f["name"] for f in folds], fontsize=9)
    ax.set_xlabel("时间点索引"); ax.set_title("滚动起源交叉验证：绿=训练，红=测试；训练窗口只增不减", fontsize=12)
    ax.grid(alpha=0.3, axis="x"); fig.tight_layout()
    figs["cv"] = savefig(fig, "ts_05_rolling_cv.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(HORIZON)
    ax.plot(x, test.values, "ko-", lw=2.2, label="真实值", zorder=5)
    colors = {"Naive(随机游走)": "#94a3b8", "Seasonal Naive": "#2563eb",
              "Ridge-AR(线性基线)": "#0891b2", "NLinear(实例归一化)": "#7c3aed",
              "DLinear(趋势-季节双头)": "#059669", "SeasonalDiff(季节差分线性)": "#d97706"}
    for name, p in preds.items():
        ax.plot(x, p, lw=1.7, color=colors.get(name, "#333"), label=name)
    ax.set_title(f"末端保留折：现代线性族 vs 真实值（最优 MASE={table[order[0]]['MASE']:.3f}）", fontsize=12)
    ax.set_xlabel("预测步（0-23 小时后）"); ax.set_ylabel("车速 (mph)")
    ax.legend(ncol=3, fontsize=8.5); ax.grid(alpha=0.3); fig.tight_layout()
    figs["forecast"] = savefig(fig, "ts_06_forecast.png")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = order
    x = np.arange(len(names)); w = 0.36
    ax[0].bar(x - w / 2, [table[k]["MAE"] for k in names], w, color="#2563eb", label="MAE")
    ax[0].bar(x + w / 2, [table[k]["RMSE"] for k in names], w, color="#7c3aed", label="RMSE")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0].set_title("末端保留折：误差指标", fontsize=12); ax[0].set_ylabel("误差 (mph)")
    ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
    ax[1].barh(x, [table[k]["MASE"] for k in names], color="#d97706")
    ax[1].axvline(1.0, color="#dc2626", ls="--", lw=1.2, label="MASE=1 朴素水平")
    ax[1].set_yticks(x); ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_title("MASE（越低越好，<1 胜过朴素）", fontsize=12); ax[1].set_xlabel("MASE")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout(); figs["metrics"] = savefig(fig, "ts_07_metrics.png")

    fig, ax = plt.subplots(1, len(order), figsize=(3.5 * len(order), 3.8), sharey=True)
    ax = np.atleast_1d(ax)
    for i, name in enumerate(order):
        plot_acf(test.values - preds[name], lags=10, ax=ax[i], color="#2563eb")
        ax[i].set_title(f"{name.split('(')[0]}\nLB p={resid_report[name]['p']:.3f}", fontsize=9)
    fig.suptitle("末端保留折残差自相关（lag≤10）：超出置信带的显著柱 = 仍有未建模结构；"
                 "标题 p 值为 CV 汇集残差的 Ljung-Box 结果", fontsize=10)
    fig.tight_layout(); figs["residual"] = savefig(fig, "ts_08_residual_acf.png")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, wts in weights.items():
        ax.plot(np.arange(len(wts)), wts, lw=1.6, marker="o", ms=3, label=label.split("(")[0])
    ax.set_xlabel("输入窗口位置（0 = 最久远，23 = 最近）")
    ax.set_ylabel("平均 |权重| 贡献")
    ax.set_title("可解释性：各输入位置对未来 24 步预测的平均贡献", fontsize=12)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3); fig.tight_layout()
    figs["weights"] = savefig(fig, "ts_09_weights.png")

    summary = {
        "n_raw": n_raw, "n_bad": n_bad, "series_len": len(ts),
        "look_back": LOOK_BACK, "horizon": HORIZON, "period": PERIOD,
        "adf_raw": adf_raw, "adf_diff": adf_diff,
        "typical_day": [round(float(v), 2) for v in typical],
        "folds": [f["name"] for f in folds],
        "cv_summary": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in cv_summary.items()},
        "holdout_metrics": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in table.items()},
        "residual_ljungbox": resid_report,
        "figures": figs,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n  [保存] results.json")
    print("\n全部完成：9 张图表 + results.json 已写入 output/")


if __name__ == "__main__":
    main()
