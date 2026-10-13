# -*- coding: utf-8 -*-
"""
学号：24108030117
阶段 3：滚动起源 CV + Seasonal Naive + Ridge-AR

规则：
- Fold 1~5 做滚动起源 CV；Holdout 使用序列最后 4 点，单独报告。
- MASE 分母只来自当前外层训练段。
- alpha 只用当前外层训练段尾部的内层验证选择。
- 内层验证的标准化统计量只来自内层拟合部分；
  外层测试集绝不参与调参/标准化。
- Ridge-AR 为直接式多步预测，一次输出未来 4 步。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 个人任务参数
# ============================================================

SID = "24108030117"

TARGET = "Gap (s)"

LOOK_BACK = 60
HORIZON = 4
PERIOD = 24

N_FOLDS = 5

# 滚动均值 / 标准差使用过去半个日周期
ROLLING_K = PERIOD // 2

# 与老师示例一致的 Ridge alpha 候选
ALPHAS = (
    0.1,
    1.0,
    10.0,
    100.0,
)

EXPECTED_N = 168

EXPECTED_ANCHOR = 0.2822


# ============================================================
# 2. 文件路径
# ============================================================

HERE = Path(__file__).resolve().parent

INPUT = (
    HERE
    / "student_output"
    / f"series_{SID}.csv"
)

OUT = (
    HERE
    / "student_output"
    / "stage3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 读取阶段 1 已验证的 168 点序列
# ============================================================

def load_series(
    path: Path
) -> pd.Series:

    if not path.exists():

        raise FileNotFoundError(
            f"找不到 {path}，"
            "请先运行阶段 1。"
        )

    df = pd.read_csv(
        path,
        parse_dates=["Date"]
    )

    required = {
        "Date",
        TARGET
    }

    if not required.issubset(
        df.columns
    ):

        raise ValueError(
            "阶段 1 输出文件缺少 "
            "Date 或 Gap (s) 字段。"
        )

    series = (
        df[
            ["Date", TARGET]
        ]
        .dropna()
        .sort_values("Date")
        .set_index("Date")[TARGET]
        .astype(float)
    )

    if len(series) != EXPECTED_N:

        raise ValueError(
            f"序列长度应为 {EXPECTED_N}，"
            f"当前为 {len(series)}。"
        )

    values = series.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "序列中存在 NaN / Inf。"
        )

    return series


# ============================================================
# 4. 滚动起源交叉验证
# ============================================================

def rolling_origin_folds(
    n: int
) -> list[dict]:

    # 第一折最小训练长度
    #
    # 60 + 4 + 24
    # = 88

    min_train = (
        LOOK_BACK
        + HORIZON
        + PERIOD
    )

    folds = []


    # --------------------------------------------------------
    # Fold 1 ~ Fold 5
    #
    # 训练窗口不断扩展
    # 测试集始终位于训练集之后
    # --------------------------------------------------------

    for k in range(
        N_FOLDS
    ):

        test_start = (
            min_train
            + k * HORIZON
        )

        test_end = (
            test_start
            + HORIZON
        )

        if test_end > n:

            raise ValueError(
                "序列太短，无法产生 "
                "5 个 CV 折。"
            )

        folds.append(
            {
                "name": (
                    f"Fold {k + 1}"
                ),

                "train": (
                    0,
                    test_start
                ),

                "test": (
                    test_start,
                    test_end
                ),
            }
        )


    # --------------------------------------------------------
    # 真正的末端保留折
    #
    # 最后 4 个点完全留到最终测试
    # --------------------------------------------------------

    holdout_start = (
        n - HORIZON
    )

    if (
        holdout_start
        < folds[-1]["test"][1]
    ):

        raise ValueError(
            "序列太短，Holdout "
            "会与 CV 测试区间重叠。"
        )

    folds.append(
        {
            "name": "Holdout",

            "train": (
                0,
                holdout_start
            ),

            "test": (
                holdout_start,
                n
            ),
        }
    )

    return folds


# ============================================================
# 5. MASE 分母
# ============================================================

def naive_scale(
    train: np.ndarray
) -> float:

    train = np.asarray(
        train,
        dtype=float
    )

    if len(train) < 2:

        raise ValueError(
            "训练段太短，"
            "无法计算 MASE。"
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
            "MASE 分母必须为正。"
        )

    return scale


# ============================================================
# 6. MAE / RMSE / MASE
# ============================================================

def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: float
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    if (
        y_true.shape
        != y_pred.shape
    ):

        raise ValueError(
            "真实值与预测值形状不一致。"
        )

    error = (
        y_true
        - y_pred
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    mase = float(
        mae / scale
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MASE": mase,
    }


# ============================================================
# 7. Seasonal Naive
# ============================================================

def seasonal_naive(
    train: np.ndarray
) -> np.ndarray:

    train = np.asarray(
        train,
        dtype=float
    )

    if len(train) < PERIOD:

        raise ValueError(
            "训练段短于 period。"
        )

    # 与任务生成器的锚点算法保持一致
    #
    # horizon = 4
    # period = 24

    return np.resize(
        train[-PERIOD:],
        HORIZON
    ).astype(float)


# ============================================================
# 8. Ridge-AR 特征
# ============================================================

def feature(
    window: np.ndarray,
    target_start: pd.Timestamp
) -> np.ndarray:

    """
    90 维因果特征：

    60 维：
        原始 look_back 窗口

    6 维：
        lag 1
        lag 2
        lag 3
        lag 24
        最近 12 步均值
        最近 12 步标准差

    24 维：
        预测起点小时 one-hot
    """

    w = np.asarray(
        window,
        dtype=float
    )

    if len(w) < max(
        PERIOD,
        ROLLING_K,
        3
    ):

        raise ValueError(
            "窗口过短。"
        )

    recent = (
        w[-ROLLING_K:]
    )


    # --------------------------------------------------------
    # 工程特征
    #
    # 全部只来自预测起点之前
    # --------------------------------------------------------

    extra = np.array(
        [
            w[-1],

            w[-2],

            w[-3],

            w[-PERIOD],

            np.mean(
                recent
            ),

            np.std(
                recent,
                ddof=0
            ),
        ],
        dtype=float
    )


    # --------------------------------------------------------
    # 小时 one-hot
    #
    # 例如 08:00：
    #
    # [0,0,...,1,...,0]
    # --------------------------------------------------------

    hour = np.zeros(
        24,
        dtype=float
    )

    hour[
        int(
            target_start.hour
        )
    ] = 1.0


    return np.concatenate(
        [
            w,
            extra,
            hour
        ]
    )


# ============================================================
# 9. 构造监督学习 X / Y
# ============================================================

def make_xy(
    train: pd.Series
) -> tuple[
    np.ndarray,
    np.ndarray
]:

    if (
        len(train)
        < LOOK_BACK + HORIZON
    ):

        raise ValueError(
            "训练段不足以构造"
            "监督学习样本。"
        )

    y = train.to_numpy(
        dtype=float
    )

    X = []
    Y = []


    # --------------------------------------------------------
    # 例如：
    #
    # 前 60 步
    #     ↓
    # 预测未来 4 步
    #
    # [xxxxxxxxxxxxxxxx] -> [yyyy]
    # --------------------------------------------------------

    for i in range(
        LOOK_BACK,
        len(y) - HORIZON + 1
    ):

        window = (
            y[
                i - LOOK_BACK:
                i
            ]
        )

        target = (
            y[
                i:
                i + HORIZON
            ]
        )

        X.append(
            feature(
                window,
                train.index[i]
            )
        )

        Y.append(
            target
        )


    return (
        np.vstack(X),
        np.vstack(Y)
    )


# ============================================================
# 10. 标准化统计量
# ============================================================

def scaler(
    X: np.ndarray,
    Y: np.ndarray
):

    x_mean = X.mean(
        axis=0
    )

    x_std = X.std(
        axis=0,
        ddof=0
    )

    y_mean = Y.mean(
        axis=0
    )

    y_std = Y.std(
        axis=0,
        ddof=0
    )


    # 防止常数特征除以 0

    x_std = np.where(
        x_std < 1e-8,
        1.0,
        x_std
    )

    y_std = np.where(
        y_std < 1e-8,
        1.0,
        y_std
    )

    return (
        x_mean,
        x_std,
        y_mean,
        y_std
    )


# ============================================================
# 11. Ridge 闭式解
# ============================================================

def ridge_fit(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float
) -> np.ndarray:

    if alpha <= 0:

        raise ValueError(
            "alpha 必须 > 0。"
        )


    # Ridge:
    #
    # W =
    # (X^T X + alpha I)^(-1) X^T Y
    #
    # 不显式求逆，使用 solve 更稳定

    identity = np.eye(
        X.shape[1]
    )

    W = np.linalg.solve(
        X.T @ X
        + alpha * identity,

        X.T @ Y
    )

    return W


# ============================================================
# 12. 内层尾部验证选择 alpha
# ============================================================

def select_alpha(
    X: np.ndarray,
    Y: np.ndarray
) -> tuple[
    float,
    pd.DataFrame
]:

    """
    只在当前外层训练集内部选择 alpha。

    最后约 20% 的监督学习样本
    作为 inner validation。

    外层 test 一眼都不能看。
    """

    n = len(X)

    n_val = min(
        max(
            HORIZON,
            n // 5
        ),
        n - 1
    )

    n_fit = (
        n - n_val
    )

    if n_fit < 2:

        raise ValueError(
            "内层拟合样本太少。"
        )


    # --------------------------------------------------------
    # 时间顺序划分
    #
    # 不能 shuffle
    # --------------------------------------------------------

    X_fit = X[:n_fit]

    Y_fit = Y[:n_fit]

    X_val = X[n_fit:]

    Y_val = Y[n_fit:]


    # --------------------------------------------------------
    # 非常重要：
    #
    # 标准化参数只由 inner-fit 得到
    #
    # inner-val 也不能参与 mean/std
    # --------------------------------------------------------

    (
        x_mean,
        x_std,
        y_mean,
        y_std
    ) = scaler(
        X_fit,
        Y_fit
    )


    X_fit_n = (
        X_fit - x_mean
    ) / x_std

    X_val_n = (
        X_val - x_mean
    ) / x_std

    Y_fit_n = (
        Y_fit - y_mean
    ) / y_std


    rows = []


    # --------------------------------------------------------
    # alpha 网格搜索
    # --------------------------------------------------------

    for alpha in ALPHAS:

        W = ridge_fit(
            X_fit_n,
            Y_fit_n,
            alpha
        )

        pred_n = (
            X_val_n @ W
        )

        pred = (
            pred_n
            * y_std
            + y_mean
        )

        val_mae = float(
            np.mean(
                np.abs(
                    Y_val
                    - pred
                )
            )
        )

        rows.append(
            {
                "alpha":
                    float(alpha),

                "val_MAE":
                    val_mae,

                "n_inner_fit":
                    n_fit,

                "n_inner_val":
                    n_val,
            }
        )


    table = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "val_MAE",
                "alpha"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    best_alpha = float(
        table.loc[
            0,
            "alpha"
        ]
    )


    return (
        best_alpha,
        table
    )


# ============================================================
# 13. Ridge-AR 拟合 + 预测
# ============================================================

def ridge_predict(
    train: pd.Series,
    target_start: pd.Timestamp
):

    # --------------------------------------------------------
    # 构造训练样本
    # --------------------------------------------------------

    X, Y = make_xy(
        train
    )


    # --------------------------------------------------------
    # 内层选择 alpha
    # --------------------------------------------------------

    (
        best_alpha,
        alpha_table
    ) = select_alpha(
        X,
        Y
    )


    # --------------------------------------------------------
    # alpha 选完以后
    #
    # 才可以使用整个“外层训练段”
    # 重新计算 mean/std 并最终拟合
    # --------------------------------------------------------

    (
        x_mean,
        x_std,
        y_mean,
        y_std
    ) = scaler(
        X,
        Y
    )


    X_n = (
        X - x_mean
    ) / x_std

    Y_n = (
        Y - y_mean
    ) / y_std


    W = ridge_fit(
        X_n,
        Y_n,
        best_alpha
    )


    # --------------------------------------------------------
    # 当前预测窗口
    # --------------------------------------------------------

    window = (
        train
        .iloc[
            -LOOK_BACK:
        ]
        .to_numpy(
            dtype=float
        )
    )


    x = feature(
        window,
        target_start
    ).reshape(
        1,
        -1
    )


    x_n = (
        x - x_mean
    ) / x_std


    pred_n = (
        x_n @ W
    )


    pred = (
        pred_n
        * y_std
        + y_mean
    )


    return (
        pred.ravel(),
        best_alpha,
        alpha_table
    )


# ============================================================
# 14. 跑全部折
# ============================================================

def run_all(
    series: pd.Series
):

    folds = rolling_origin_folds(
        len(series)
    )

    metric_rows = []

    prediction_rows = []

    alpha_rows = []


    for order, fold in enumerate(
        folds,
        start=1
    ):

        train_start, train_end = (
            fold["train"]
        )

        test_start, test_end = (
            fold["test"]
        )


        train = series.iloc[
            train_start:
            train_end
        ]

        test = series.iloc[
            test_start:
            test_end
        ]


        y_train = train.to_numpy(
            dtype=float
        )

        y_test = test.to_numpy(
            dtype=float
        )


        # ----------------------------------------------------
        # 当前折 MASE 分母
        # ----------------------------------------------------

        scale = naive_scale(
            y_train
        )


        # ----------------------------------------------------
        # Seasonal Naive
        # ----------------------------------------------------

        sn_pred = seasonal_naive(
            y_train
        )

        sn_metrics = metrics(
            y_test,
            sn_pred,
            scale
        )


        metric_rows.append(
            {
                "fold":
                    fold["name"],

                "order":
                    order,

                "model":
                    "Seasonal Naive",

                "train_n":
                    len(train),

                "alpha":
                    np.nan,

                **sn_metrics,
            }
        )


        # ----------------------------------------------------
        # Ridge-AR
        # ----------------------------------------------------

        (
            ridge_pred,
            alpha,
            alpha_table
        ) = ridge_predict(
            train,
            test.index[0]
        )


        ridge_metrics = metrics(
            y_test,
            ridge_pred,
            scale
        )


        metric_rows.append(
            {
                "fold":
                    fold["name"],

                "order":
                    order,

                "model":
                    "Ridge-AR",

                "train_n":
                    len(train),

                "alpha":
                    alpha,

                **ridge_metrics,
            }
        )


        # ----------------------------------------------------
        # 保存 alpha 搜索过程
        # ----------------------------------------------------

        for _, row in (
            alpha_table.iterrows()
        ):

            alpha_rows.append(
                {
                    "fold":
                        fold["name"],

                    "order":
                        order,

                    "alpha":
                        float(
                            row[
                                "alpha"
                            ]
                        ),

                    "inner_val_MAE":
                        float(
                            row[
                                "val_MAE"
                            ]
                        ),

                    "n_inner_fit":
                        int(
                            row[
                                "n_inner_fit"
                            ]
                        ),

                    "n_inner_val":
                        int(
                            row[
                                "n_inner_val"
                            ]
                        ),

                    "selected":
                        bool(
                            float(
                                row[
                                    "alpha"
                                ]
                            )
                            == alpha
                        ),
                }
            )


        # ----------------------------------------------------
        # 保存每个 horizon 的预测
        # ----------------------------------------------------

        for h in range(
            HORIZON
        ):

            prediction_rows.append(
                {
                    "fold":
                        fold["name"],

                    "order":
                        order,

                    "horizon":
                        h + 1,

                    "timestamp":
                        test.index[h],

                    "actual":
                        float(
                            y_test[h]
                        ),

                    "Seasonal_Naive":
                        float(
                            sn_pred[h]
                        ),

                    "Ridge_AR":
                        float(
                            ridge_pred[h]
                        ),
                }
            )


    metrics_df = pd.DataFrame(
        metric_rows
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    alpha_df = pd.DataFrame(
        alpha_rows
    )


    return (
        folds,
        metrics_df,
        predictions_df,
        alpha_df
    )


# ============================================================
# 15. CV 均值 ± 标准差
# ============================================================

def summary_table(
    metrics_df: pd.DataFrame
) -> pd.DataFrame:

    # Holdout 不参加 CV 平均
    cv = metrics_df[
        metrics_df["fold"]
        != "Holdout"
    ]


    rows = []


    for model, group in (
        cv.groupby("model")
    ):

        holdout = metrics_df[
            (
                metrics_df["fold"]
                == "Holdout"
            )
            &
            (
                metrics_df["model"]
                == model
            )
        ].iloc[0]


        rows.append(
            {
                "model":
                    model,

                "CV_MAE_mean":
                    group[
                        "MAE"
                    ].mean(),

                "CV_MAE_std":
                    group[
                        "MAE"
                    ].std(
                        ddof=0
                    ),

                "CV_RMSE_mean":
                    group[
                        "RMSE"
                    ].mean(),

                "CV_RMSE_std":
                    group[
                        "RMSE"
                    ].std(
                        ddof=0
                    ),

                "CV_MASE_mean":
                    group[
                        "MASE"
                    ].mean(),

                "CV_MASE_std":
                    group[
                        "MASE"
                    ].std(
                        ddof=0
                    ),

                "Holdout_MASE":
                    holdout[
                        "MASE"
                    ],
            }
        )


    return (
        pd.DataFrame(rows)
        .sort_values(
            "CV_MASE_mean"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 16. 保存图表
# ============================================================

def save_plots(
    folds: list[dict],
    predictions: pd.DataFrame,
    n: int
):

    # --------------------------------------------------------
    # 图 1：滚动起源划分
    # --------------------------------------------------------

    path1 = (
        OUT
        / "04_rolling_origin_cv.png"
    )


    fig, ax = plt.subplots(
        figsize=(11, 4.8)
    )


    for i, fold in enumerate(
        folds
    ):

        train_start, train_end = (
            fold["train"]
        )

        test_start, test_end = (
            fold["test"]
        )


        ax.barh(
            i,

            train_end
            - train_start,

            left=train_start,

            height=0.5,

            color="tab:blue",

            label=(
                "Train"
                if i == 0
                else None
            )
        )


        ax.barh(
            i,

            test_end
            - test_start,

            left=test_start,

            height=0.5,

            color="tab:orange",

            label=(
                "Test"
                if i == 0
                else None
            )
        )


    ax.set_yticks(
        range(
            len(folds)
        )
    )

    ax.set_yticklabels(
        [
            f["name"]
            for f in folds
        ]
    )

    ax.set_xlim(
        0,
        n
    )

    ax.set_xlabel(
        "Time-step index"
    )

    ax.set_ylabel(
        "Split"
    )

    ax.set_title(
        "Rolling-origin cross-validation"
    )

    ax.legend()

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        path1,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


    # --------------------------------------------------------
    # 图 2：六折预测
    # --------------------------------------------------------

    path2 = (
        OUT
        / "05_fold_forecasts_ridge_vs_seasonal.png"
    )


    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 7)
    )

    axes = (
        axes.ravel()
    )


    fold_names = (
        predictions[
            "fold"
        ]
        .drop_duplicates()
    )


    for ax, name in zip(
        axes,
        fold_names
    ):

        group = predictions[
            predictions["fold"]
            == name
        ]

        x = group[
            "horizon"
        ]


        ax.plot(
            x,
            group["actual"],
            marker="o",
            label="Actual"
        )

        ax.plot(
            x,
            group[
                "Seasonal_Naive"
            ],
            marker="o",
            label="Seasonal Naive"
        )

        ax.plot(
            x,
            group[
                "Ridge_AR"
            ],
            marker="o",
            label="Ridge-AR"
        )


        ax.set_title(
            name
        )

        ax.set_xlabel(
            "Forecast horizon (hour)"
        )

        ax.set_ylabel(
            "Gap (s)"
        )

        ax.set_xticks(
            range(
                1,
                HORIZON + 1
            )
        )

        ax.grid(
            alpha=0.25
        )


    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )


    fig.suptitle(
        "Fold forecasts: Ridge-AR vs Seasonal Naive",
        y=0.995
    )


    fig.legend(
        handles,
        labels,

        loc="upper center",

        bbox_to_anchor=(
            0.5,
            0.955
        ),

        ncol=3
    )


    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.90
        )
    )


    fig.savefig(
        path2,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


    return (
        path1,
        path2
    )


# ============================================================
# 17. 主程序
# ============================================================

def main():

    series = load_series(
        INPUT
    )


    (
        folds,
        metrics_df,
        predictions_df,
        alpha_df
    ) = run_all(
        series
    )


    # --------------------------------------------------------
    # 再次校验折 1 锚点
    # --------------------------------------------------------

    anchor = float(
        metrics_df[
            (
                metrics_df["fold"]
                == "Fold 1"
            )
            &
            (
                metrics_df["model"]
                == "Seasonal Naive"
            )
        ][
            "MASE"
        ].iloc[0]
    )


    assert (
        round(
            anchor,
            4
        )
        == EXPECTED_ANCHOR
    ), (
        f"锚点不一致："
        f"当前 {anchor:.4f}，"
        f"任务书 {EXPECTED_ANCHOR:.4f}"
    )


    print(
        "=" * 78
    )

    print(
        "阶段 3："
        "Rolling-origin CV + "
        "Seasonal Naive + Ridge-AR"
    )

    print(
        "=" * 78
    )


    print(
        "Fold 1 Seasonal Naive "
        f"MASE = {anchor:.10f} "
        f"-> {anchor:.4f}  ✅"
    )


    # --------------------------------------------------------
    # 输出每折结果
    # --------------------------------------------------------

    for fold in folds:

        train_start, train_end = (
            fold["train"]
        )

        test_start, test_end = (
            fold["test"]
        )


        print()

        print(
            f"{fold['name']}: "
            f"train=[{train_start}:{train_end}) "
            f"n={train_end - train_start}, "
            f"test=[{test_start}:{test_end}) "
            f"n={test_end - test_start}"
        )


        rows = metrics_df[
            metrics_df["fold"]
            == fold["name"]
        ]


        for _, row in (
            rows.iterrows()
        ):

            if pd.isna(
                row["alpha"]
            ):

                alpha_text = "-"

            else:

                alpha_text = (
                    f"{row['alpha']:.1f}"
                )


            print(
                f"  {row['model']:<16} "
                f"alpha={alpha_text:<5} "
                f"MAE={row['MAE']:.6f} "
                f"RMSE={row['RMSE']:.6f} "
                f"MASE={row['MASE']:.6f}"
            )


        chosen = alpha_df[
            (
                alpha_df["fold"]
                == fold["name"]
            )
            &
            (
                alpha_df["selected"]
            )
        ].iloc[0]


        print(
            "  selected alpha="
            f"{chosen['alpha']:.1f}; "
            "inner fit/val="
            f"{int(chosen['n_inner_fit'])}/"
            f"{int(chosen['n_inner_val'])}"
        )


    # --------------------------------------------------------
    # CV 汇总
    # --------------------------------------------------------

    summary_df = summary_table(
        metrics_df
    )


    print()

    print(
        "=" * 78
    )

    print(
        "CV summary "
        "(Fold 1~5 only; "
        "Holdout separate)"
    )

    print(
        "=" * 78
    )


    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"{row['model']:<16} "
            f"CV MASE="
            f"{row['CV_MASE_mean']:.6f} "
            f"± "
            f"{row['CV_MASE_std']:.6f}  "

            f"CV MAE="
            f"{row['CV_MAE_mean']:.6f} "
            f"± "
            f"{row['CV_MAE_std']:.6f}  "

            f"Holdout MASE="
            f"{row['Holdout_MASE']:.6f}"
        )


    # --------------------------------------------------------
    # 保存结果 CSV
    # --------------------------------------------------------

    metrics_path = (
        OUT
        / "fold_metrics.csv"
    )

    predictions_path = (
        OUT
        / "fold_predictions.csv"
    )

    alpha_path = (
        OUT
        / "alpha_search.csv"
    )

    summary_path = (
        OUT
        / "cv_summary.csv"
    )


    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig"
    )

    alpha_df.to_csv(
        alpha_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # 保存图
    # --------------------------------------------------------

    path1, path2 = save_plots(
        folds,
        predictions_df,
        len(series)
    )


    print()

    print(
        "[保存]",
        metrics_path
    )

    print(
        "[保存]",
        predictions_path
    )

    print(
        "[保存]",
        alpha_path
    )

    print(
        "[保存]",
        summary_path
    )

    print(
        "[保存]",
        path1
    )

    print(
        "[保存]",
        path2
    )


    print()

    print(
        "✅ 阶段 3 完成。"
        "下一步：NLinear。"
    )


# ============================================================
# 18. 程序入口
# ============================================================

if __name__ == "__main__":
    main()