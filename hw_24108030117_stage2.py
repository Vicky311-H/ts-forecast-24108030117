# -*- coding: utf-8 -*-
"""
学号：24108030117
时间序列实验 —— 阶段 2：EDA + ACF + 第 1 折 Seasonal Naive 锚点校验

前提：阶段 1 已通过，并生成：
student_output/series_24108030117.csv

固定参数：
look_back = 60
horizon   = 4
period    = 24
第 1 折 Seasonal Naive MASE 锚点 = 0.2822
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


# ============================================================
# 1. 固定参数
# ============================================================

SID = "24108030117"
TARGET_COL = "Gap (s)"

LOOK_BACK = 60
HORIZON = 4
PERIOD = 24

EXPECTED_N_POINTS = 168
EXPECTED_FOLD1_MASE = 0.2822

HERE = Path(__file__).resolve().parent

INPUT_PATH = (
    HERE
    / "student_output"
    / f"series_{SID}.csv"
)

OUTPUT_DIR = (
    HERE
    / "student_output"
    / "stage2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 读取阶段 1 已验证的 168 点序列
# ============================================================

def load_series(csv_path: Path) -> pd.Series:

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到阶段 1 输出文件：{csv_path}\n"
            "请先运行 hw_24108030117.py，"
            "确认正式检查点 1 已通过。"
        )

    df = pd.read_csv(
        csv_path,
        parse_dates=["Date"]
    )

    required = {
        "Date",
        TARGET_COL
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"序列文件缺少字段：{sorted(missing)}"
        )

    series = (
        df[
            ["Date", TARGET_COL]
        ]
        .dropna()
        .sort_values("Date")
        .set_index("Date")[TARGET_COL]
        .astype(float)
    )

    if len(series) != EXPECTED_N_POINTS:
        raise ValueError(
            f"序列长度错误：当前 {len(series)}，"
            f"应为 {EXPECTED_N_POINTS}。"
        )

    values = series.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "序列中存在 NaN 或无穷值。"
        )

    return series


# ============================================================
# 3. 误差指标
# ============================================================

def calc_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def calc_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def calc_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train: np.ndarray
) -> tuple[float, float]:

    """
    按任务书要求计算 MASE。

    注意：
    MASE 分母只能使用当前折的训练数据。

    scale =
    mean(|train[t] - train[t-1]|)
    """

    if len(train) < 2:
        raise ValueError(
            "训练段至少需要 2 个点"
            "才能计算 MASE。"
        )

    scale = float(
        np.mean(
            np.abs(
                np.diff(train)
            )
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError(
            "MASE 分母无效。"
        )

    mae = calc_mae(
        y_true,
        y_pred
    )

    mase = mae / scale

    return float(mase), scale


# ============================================================
# 4. 第 1 折 Seasonal Naive
# ============================================================

def run_fold1_seasonal_naive(
    series: pd.Series
) -> dict:

    y = series.to_numpy(
        dtype=float
    )

    # 老师任务生成器中的锚点算法：
    #
    # min_train =
    # look_back + horizon + period
    #
    # 60 + 4 + 24 = 88

    min_train = (
        LOOK_BACK
        + HORIZON
        + PERIOD
    )

    if len(y) < (
        min_train
        + HORIZON
    ):
        raise ValueError(
            "序列长度不足。"
        )

    # 第 1 折训练集
    train = y[
        :min_train
    ]

    # 第 1 折测试集
    test = y[
        min_train:
        min_train + HORIZON
    ]

    # --------------------------------------------------------
    # Seasonal Naive
    #
    # 使用上一个完整周期中
    # 与未来位置相对应的数据作为预测
    #
    # 这里与老师的：
    #
    # np.resize(tr[-P:], H)
    #
    # 完全一致
    # --------------------------------------------------------

    pred = np.resize(
        train[-PERIOD:],
        HORIZON
    ).astype(float)

    mae = calc_mae(
        test,
        pred
    )

    rmse = calc_rmse(
        test,
        pred
    )

    mase, mase_scale = calc_mase(
        test,
        pred,
        train
    )

    return {
        "min_train": min_train,
        "train": train,
        "test": test,
        "pred": pred,
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "mase_scale": mase_scale
    }


# ============================================================
# 5. EDA：时间序列图
# ============================================================

def save_series_plot(
    series: pd.Series
) -> Path:

    path = (
        OUTPUT_DIR
        / "01_hourly_gap_series.png"
    )

    y = series.to_numpy(
        dtype=float
    )

    x = np.arange(
        len(y)
    )

    fig, ax = plt.subplots(
        figsize=(12, 4.8)
    )

    ax.plot(
        x,
        y,
        linewidth=1.5
    )

    ax.set_title(
        "Hourly Gap Time Series"
    )

    ax.set_xlabel(
        "Time-step index"
    )

    ax.set_ylabel(
        "Gap (s)"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# 6. ACF 图
# ============================================================

def save_acf_plot(
    series: pd.Series
) -> tuple[Path, np.ndarray]:

    path = (
        OUTPUT_DIR
        / "02_acf.png"
    )

    y = series.to_numpy(
        dtype=float
    )

    # 观察两个完整日周期
    max_lag = min(
        48,
        len(y) - 1
    )

    acf_values = acf(
        y,
        nlags=max_lag,
        fft=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 4.8)
    )

    plot_acf(
        y,
        lags=max_lag,
        alpha=0.05,
        zero=True,
        ax=ax
    )

    # 标记一天的周期位置
    ax.axvline(
        PERIOD,
        linestyle="--",
        linewidth=1.2,
        label=f"Daily period = {PERIOD}"
    )

    ax.set_title(
        "Autocorrelation Function"
    )

    ax.set_xlabel(
        "Lag (hours)"
    )

    ax.set_ylabel(
        "ACF"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    return (
        path,
        acf_values
    )


# ============================================================
# 7. Fold 1 预测图
# ============================================================

def save_fold1_plot(
    result: dict
) -> Path:

    path = (
        OUTPUT_DIR
        / "03_fold1_seasonal_naive.png"
    )

    test = result[
        "test"
    ]

    pred = result[
        "pred"
    ]

    mase = result[
        "mase"
    ]

    x = np.arange(
        1,
        HORIZON + 1
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    ax.plot(
        x,
        test,
        marker="o",
        linewidth=1.8,
        label="Actual"
    )

    ax.plot(
        x,
        pred,
        marker="o",
        linewidth=1.8,
        label="Seasonal Naive"
    )

    ax.set_title(
        "Fold 1 Seasonal Naive Forecast "
        f"(MASE = {mase:.4f})"
    )

    ax.set_xlabel(
        "Forecast horizon (hour)"
    )

    ax.set_ylabel(
        "Gap (s)"
    )

    ax.set_xticks(x)

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# 8. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 读取阶段 1 序列
    # --------------------------------------------------------

    series = load_series(
        INPUT_PATH
    )

    print(
        "=" * 70
    )

    print(
        "阶段 2：EDA + ACF + Fold 1 Seasonal Naive"
    )

    print(
        "=" * 70
    )

    print(
        f"序列长度：{len(series)}"
    )

    print(
        f"look_back = {LOOK_BACK}"
    )

    print(
        f"horizon   = {HORIZON}"
    )

    print(
        f"period    = {PERIOD}"
    )


    # --------------------------------------------------------
    # 时间序列图
    # --------------------------------------------------------

    series_plot = save_series_plot(
        series
    )

    print()

    print(
        "[保存]",
        series_plot
    )


    # --------------------------------------------------------
    # ACF
    # --------------------------------------------------------

    (
        acf_plot,
        acf_values
    ) = save_acf_plot(
        series
    )

    print(
        "[保存]",
        acf_plot
    )

    print(
        f"ACF(lag={PERIOD}) "
        f"= {acf_values[PERIOD]:.6f}"
    )

    if len(acf_values) > (
        2 * PERIOD
    ):

        print(
            f"ACF(lag={2 * PERIOD}) "
            f"= "
            f"{acf_values[2 * PERIOD]:.6f}"
        )


    # --------------------------------------------------------
    # 查看 ACF 最大的若干 lag
    # --------------------------------------------------------

    lag_order = (
        np.argsort(
            acf_values[1:]
        )[::-1]
        + 1
    )

    top_lags = lag_order[
        :8
    ]

    print()

    print(
        "ACF 较大的 lag："
    )

    for lag in top_lags:

        print(
            f"  lag={int(lag):2d}, "
            f"ACF={acf_values[lag]:.6f}"
        )


    # --------------------------------------------------------
    # Fold 1 Seasonal Naive
    # --------------------------------------------------------

    result = (
        run_fold1_seasonal_naive(
            series
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "正式检查点 2："
        "Fold 1 Seasonal Naive"
    )

    print(
        "=" * 70
    )

    print(
        "最小训练长度 =",
        result["min_train"]
    )

    print(
        "训练段长度   =",
        len(
            result["train"]
        )
    )

    print(
        "测试段长度   =",
        len(
            result["test"]
        )
    )

    print(
        "MASE 分母    = "
        f"{result['mase_scale']:.10f}"
    )


    # --------------------------------------------------------
    # 打印真实值
    # --------------------------------------------------------

    print()

    print(
        "真实值："
    )

    for i, value in enumerate(
        result["test"],
        start=1
    ):

        print(
            f"  t+{i}: "
            f"{value:.10f}"
        )


    # --------------------------------------------------------
    # 打印预测值
    # --------------------------------------------------------

    print()

    print(
        "Seasonal Naive 预测值："
    )

    for i, value in enumerate(
        result["pred"],
        start=1
    ):

        print(
            f"  t+{i}: "
            f"{value:.10f}"
        )


    # --------------------------------------------------------
    # 指标
    # --------------------------------------------------------

    print()

    print(
        "Fold 1 MAE  = "
        f"{result['mae']:.10f}"
    )

    print(
        "Fold 1 RMSE = "
        f"{result['rmse']:.10f}"
    )

    print(
        "Fold 1 MASE = "
        f"{result['mase']:.10f}"
    )

    print(
        "四舍五入后  = "
        f"{result['mase']:.4f}"
    )


    # --------------------------------------------------------
    # 硬性锚点校验
    # --------------------------------------------------------

    if (
        round(
            result["mase"],
            4
        )
        != EXPECTED_FOLD1_MASE
    ):

        raise AssertionError(
            "\n❌ Fold 1 Seasonal Naive "
            "MASE 锚点不一致！"
            f"\n当前："
            f"{result['mase']:.4f}"
            f"\n任务书："
            f"{EXPECTED_FOLD1_MASE:.4f}"
        )


    # --------------------------------------------------------
    # Fold 1 图
    # --------------------------------------------------------

    fold1_plot = save_fold1_plot(
        result
    )

    print(
        "[保存]",
        fold1_plot
    )


    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print()

    print(
        "✅ 正式检查点 2 通过！"
    )

    print(
        "Fold 1 Seasonal Naive MASE "
        f"= {EXPECTED_FOLD1_MASE:.4f}"
    )

    print()

    print(
        "下一阶段："
        "滚动起源 CV + Ridge-AR"
    )


# ============================================================
# 9. 程序入口
# ============================================================

if __name__ == "__main__":
    main()