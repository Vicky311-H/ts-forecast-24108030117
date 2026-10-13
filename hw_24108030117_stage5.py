# -*- coding: utf-8 -*-
"""
学号：24108030117
阶段 5：逐折 Ljung-Box 残差诊断 + 数据泄露反事实实验

说明：
1. Ljung-Box 必须逐折做，绝不把不同折的残差拼接后再检验。
2. 每折只有 horizon=4 个残差，因此只检验 lag=1；
   样本极小，p 值只能作为弱证据。
3. 反事实实验故意制造“全序列标准化统计量泄露”。
   这是错误做法，只用于报告中的泄露反事实，
   不能作为最终模型结果。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# 复用阶段 3
# ============================================================

from hw_24108030117_stage3 import (
    HORIZON,
    LOOK_BACK,
    N_FOLDS,
    INPUT,
    EXPECTED_ANCHOR,
    load_series,
    rolling_origin_folds,
    naive_scale,
    metrics,
    scaler,
    ridge_fit,
    select_alpha,
)


# ============================================================
# 复用阶段 4
# ============================================================

from hw_24108030117_stage4 import (
    run_three_models,
    make_nlinear_xy,
    nlinear_feature,
)


# ============================================================
# 1. 输出目录
# ============================================================

HERE = Path(__file__).resolve().parent

OUT = (
    HERE
    / "student_output"
    / "stage5"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# 三模型在预测结果表中的列名
MODEL_COLUMNS = {

    "Seasonal Naive":
        "Seasonal_Naive",

    "Ridge-AR":
        "Ridge_AR",

    "NLinear":
        "NLinear",
}


# ============================================================
# 2. 单折 Ljung-Box 检验
# ============================================================

def ljung_box_one_fold(
    residuals: np.ndarray
) -> tuple[int, float, float]:

    """
    对“一折内部”的残差做 Ljung-Box。

    本实验：
    horizon = 4

    所以每折只有 4 个残差。

    lag 必须明显小于 n，
    因此这里只使用 lag=1。
    """

    r = np.asarray(
        residuals,
        dtype=float
    )


    # --------------------------------------------------------
    # 基础检查
    # --------------------------------------------------------

    if r.ndim != 1:

        raise ValueError(
            "residuals 必须是一维数组。"
        )


    if len(r) < 3:

        raise ValueError(
            "残差少于 3 个，"
            "不适合做本实验的 Ljung-Box。"
        )


    if not np.isfinite(
        r
    ).all():

        raise ValueError(
            "残差中存在 NaN / Inf。"
        )


    # --------------------------------------------------------
    # 当前任务固定 lag=1
    # --------------------------------------------------------

    lag = 1


    # --------------------------------------------------------
    # 边界情况：
    #
    # 如果所有残差完全一样，
    # 没有正常可估计的自相关变化。
    # --------------------------------------------------------

    if np.std(
        r,
        ddof=0
    ) < 1e-12:

        return (
            lag,
            0.0,
            1.0
        )


    # --------------------------------------------------------
    # Ljung-Box
    #
    # H0：
    # residuals 在给定 lag 下没有自相关
    #
    # p < 0.05：
    # 拒绝 H0，说明还有显著结构
    #
    # p >= 0.05：
    # 未拒绝 H0
    # --------------------------------------------------------

    result = acorr_ljungbox(
        r,
        lags=[lag],
        model_df=0,
        return_df=True,
    )


    stat = float(
        result.loc[
            lag,
            "lb_stat"
        ]
    )


    pvalue = float(
        result.loc[
            lag,
            "lb_pvalue"
        ]
    )


    return (
        lag,
        stat,
        pvalue
    )


# ============================================================
# 3. 所有模型逐折残差诊断
# ============================================================

def residual_diagnostics(
    predictions: pd.DataFrame
) -> pd.DataFrame:

    rows = []


    fold_names = (
        predictions[
            "fold"
        ]
        .drop_duplicates()
        .tolist()
    )


    # --------------------------------------------------------
    # 每折独立处理
    #
    # 特别注意：
    # 这里绝不 concat 不同折的 residual
    # --------------------------------------------------------

    for fold_order, fold_name in enumerate(
        fold_names,
        start=1
    ):

        g = (
            predictions[
                predictions["fold"]
                == fold_name
            ]
            .sort_values(
                "horizon"
            )
            .reset_index(
                drop=True
            )
        )


        actual = (
            g[
                "actual"
            ]
            .to_numpy(
                dtype=float
            )
        )


        if len(
            actual
        ) != HORIZON:

            raise ValueError(
                f"{fold_name} 应有 "
                f"{HORIZON} 个残差，"
                f"当前为 {len(actual)}。"
            )


        # ----------------------------------------------------
        # 三个模型分别检验
        # ----------------------------------------------------

        for (
            model_name,
            pred_col
        ) in MODEL_COLUMNS.items():


            pred = (
                g[
                    pred_col
                ]
                .to_numpy(
                    dtype=float
                )
            )


            # residual = actual - prediction
            residuals = (
                actual - pred
            )


            (
                lag,
                stat,
                pvalue
            ) = ljung_box_one_fold(
                residuals
            )


            rows.append(
                {
                    "fold":
                        fold_name,

                    "order":
                        fold_order,

                    "model":
                        model_name,

                    "n_residuals":
                        len(
                            residuals
                        ),

                    "lag":
                        lag,

                    "lb_stat":
                        stat,

                    "lb_pvalue":
                        pvalue,

                    "reject_white_noise_0.05":
                        bool(
                            pvalue < 0.05
                        ),

                    "residual_h1":
                        float(
                            residuals[0]
                        ),

                    "residual_h2":
                        float(
                            residuals[1]
                        ),

                    "residual_h3":
                        float(
                            residuals[2]
                        ),

                    "residual_h4":
                        float(
                            residuals[3]
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# 4. 构造故意泄露的全序列标准化器
# ============================================================

def build_leaky_full_scaler(
    full_series: pd.Series
):

    """
    【故意错误】

    使用完整 168 点序列来计算
    NLinear 的 X/Y 标准化统计量。

    其中包含未来测试折的信息。

    因此：
    这是数据泄露。

    只用于反事实实验。
    """


    # --------------------------------------------------------
    # 完整序列：
    #
    # 这里会构造包含未来数据的 X / Y
    # --------------------------------------------------------

    X_full, Y_full = (
        make_nlinear_xy(
            full_series
        )
    )


    # --------------------------------------------------------
    # 故意用全序列计算 mean/std
    #
    # 【泄露发生在这里】
    # --------------------------------------------------------

    return scaler(
        X_full,
        Y_full
    )


# ============================================================
# 5. 使用泄露标准化器的 NLinear
# ============================================================

def nlinear_predict_with_leaky_scaler(
    train: pd.Series,
    target_start: pd.Timestamp,
    leaky_scaler
):

    """
    为了只观察“标准化泄露”的影响：

    alpha：
        仍然严格只使用当前训练段选择

    模型训练数据：
        仍然只使用当前训练段

    唯一故意做错的地方：
        mean/std 来自完整 168 点序列
    """


    # --------------------------------------------------------
    # 当前训练段监督学习数据
    # --------------------------------------------------------

    X, Y = make_nlinear_xy(
        train
    )


    # --------------------------------------------------------
    # alpha 仍然正确：
    #
    # 只使用当前训练数据内部验证
    #
    # 所以不会混淆两种泄露
    # --------------------------------------------------------

    (
        best_alpha,
        _
    ) = select_alpha(
        X,
        Y
    )


    # --------------------------------------------------------
    # 【故意泄露】
    #
    # mean / std 来自整个序列
    # --------------------------------------------------------

    (
        x_mu,
        x_sd,
        y_mu,
        y_sd
    ) = leaky_scaler


    # --------------------------------------------------------
    # 使用泄露统计量标准化训练数据
    # --------------------------------------------------------

    Xn = (
        X - x_mu
    ) / x_sd


    Yn = (
        Y - y_mu
    ) / y_sd


    # --------------------------------------------------------
    # Ridge 线性层
    # --------------------------------------------------------

    W = ridge_fit(
        Xn,
        Yn,
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


    feat, base = (
        nlinear_feature(
            window,
            target_start
        )
    )


    # --------------------------------------------------------
    # 同样使用泄露标准化器
    # --------------------------------------------------------

    xn = (
        feat.reshape(
            1,
            -1
        )
        - x_mu
    ) / x_sd


    # --------------------------------------------------------
    # 相对值预测
    # --------------------------------------------------------

    pred_relative = (
        xn @ W
    )


    pred_relative = (
        pred_relative
        * y_sd
        + y_mu
    )


    # --------------------------------------------------------
    # NLinear 加回窗口末值
    # --------------------------------------------------------

    pred = (
        pred_relative
        .ravel()
        + base
    )


    return (
        pred,
        best_alpha
    )


# ============================================================
# 6. 泄露反事实实验
# ============================================================

def leakage_counterfactual(
    series: pd.Series,
    correct_metrics: pd.DataFrame
) -> pd.DataFrame:

    folds = rolling_origin_folds(
        len(series)
    )


    # --------------------------------------------------------
    # 注意：
    #
    # 这个 scaler 故意提前看完整 168 点
    # --------------------------------------------------------

    leaky_scaler = (
        build_leaky_full_scaler(
            series
        )
    )


    rows = []


    for order, f in enumerate(
        folds,
        start=1
    ):

        tr0, tr1 = (
            f["train"]
        )

        te0, te1 = (
            f["test"]
        )


        train = series.iloc[
            tr0:
            tr1
        ]


        test = series.iloc[
            te0:
            te1
        ]


        y_train = (
            train
            .to_numpy(
                dtype=float
            )
        )


        y_test = (
            test
            .to_numpy(
                dtype=float
            )
        )


        # ----------------------------------------------------
        # MASE 分母仍然保持正确：
        #
        # 只能来自当前训练段
        # ----------------------------------------------------

        scale = naive_scale(
            y_train
        )


        # ----------------------------------------------------
        # 泄露版本预测
        # ----------------------------------------------------

        (
            leaky_pred,
            alpha
        ) = (
            nlinear_predict_with_leaky_scaler(
                train,
                test.index[0],
                leaky_scaler
            )
        )


        leaky_m = metrics(
            y_test,
            leaky_pred,
            scale
        )


        # ----------------------------------------------------
        # 找到阶段 4 中正确的 NLinear MASE
        # ----------------------------------------------------

        correct_row = (
            correct_metrics[
                (
                    correct_metrics["fold"]
                    == f["name"]
                )
                &
                (
                    correct_metrics["model"]
                    == "NLinear"
                )
            ]
        )


        if len(
            correct_row
        ) != 1:

            raise ValueError(
                f"找不到 {f['name']} "
                "的唯一 NLinear 正确结果。"
            )


        correct_mase = float(
            correct_row
            .iloc[0][
                "MASE"
            ]
        )


        rows.append(
            {
                "fold":
                    f["name"],

                "order":
                    order,

                "alpha":
                    alpha,

                "correct_MASE":
                    correct_mase,

                "leaky_MASE":
                    float(
                        leaky_m[
                            "MASE"
                        ]
                    ),

                "MASE_change_leaky_minus_correct":
                    (
                        float(
                            leaky_m[
                                "MASE"
                            ]
                        )
                        - correct_mase
                    ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# 7. 图：逐折 Ljung-Box p-value
# ============================================================

def save_ljungbox_plot(
    lb_df: pd.DataFrame
) -> Path:

    path = (
        OUT
        / "08_ljungbox_pvalues_by_fold.png"
    )


    fig, ax = plt.subplots(
        figsize=(
            10.5,
            5.0
        )
    )


    # --------------------------------------------------------
    # 三模型各一条线
    # --------------------------------------------------------

    for model, g in (
        lb_df.groupby(
            "model"
        )
    ):

        g = g.sort_values(
            "order"
        )


        ax.plot(
            g[
                "order"
            ],

            g[
                "lb_pvalue"
            ],

            marker="o",

            label=model
        )


    # --------------------------------------------------------
    # 显著性水平
    # --------------------------------------------------------

    ax.axhline(
        0.05,
        linestyle="--",
        linewidth=1.0,
        label="Significance level = 0.05"
    )


    ax.set_title(
        "Fold-wise Ljung-Box residual test (lag = 1)"
    )


    ax.set_xlabel(
        "Fold"
    )


    ax.set_ylabel(
        "Ljung-Box p-value"
    )


    ax.set_xticks(
        range(
            1,
            N_FOLDS + 2
        )
    )


    ax.set_xticklabels(
        [
            *[
                f"Fold {i}"
                for i in range(
                    1,
                    N_FOLDS + 1
                )
            ],

            "Holdout",
        ]
    )


    ax.set_ylim(
        0.0,
        1.0
    )


    ax.grid(
        alpha=0.25
    )


    ax.legend()


    fig.tight_layout()


    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    return path


# ============================================================
# 8. 图：正确 NLinear vs 泄露 NLinear
# ============================================================

def save_leakage_plot(
    leak_df: pd.DataFrame
) -> Path:

    path = (
        OUT
        / "09_nlinear_leakage_counterfactual.png"
    )


    fig, ax = plt.subplots(
        figsize=(
            10.5,
            5.0
        )
    )


    x = leak_df[
        "order"
    ].to_numpy()


    ax.plot(
        x,
        leak_df[
            "correct_MASE"
        ],
        marker="o",
        label="Correct NLinear"
    )


    ax.plot(
        x,
        leak_df[
            "leaky_MASE"
        ],
        marker="o",
        label="Leaky full-series scaling"
    )


    ax.set_title(
        "Leakage counterfactual: correct vs leaky NLinear"
    )


    ax.set_xlabel(
        "Fold"
    )


    ax.set_ylabel(
        "MASE"
    )


    ax.set_xticks(
        range(
            1,
            N_FOLDS + 2
        )
    )


    ax.set_xticklabels(
        [
            *[
                f"Fold {i}"
                for i in range(
                    1,
                    N_FOLDS + 1
                )
            ],

            "Holdout",
        ]
    )


    ax.grid(
        alpha=0.25
    )


    ax.legend()


    fig.tight_layout()


    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    return path


# ============================================================
# 9. 主程序
# ============================================================

def main():

    series = load_series(
        INPUT
    )


    # --------------------------------------------------------
    # 重新获得阶段 4 的三模型结果
    # --------------------------------------------------------

    (
        folds,
        metric_df,
        pred_df,
        _
    ) = run_three_models(
        series
    )


    # --------------------------------------------------------
    # 再次检查最重要的任务锚点
    # --------------------------------------------------------

    anchor = float(
        metric_df[
            (
                metric_df["fold"]
                == "Fold 1"
            )
            &
            (
                metric_df["model"]
                == "Seasonal Naive"
            )
        ][
            "MASE"
        ]
        .iloc[0]
    )


    assert (
        round(
            anchor,
            4
        )
        == EXPECTED_ANCHOR
    )


    # ========================================================
    # A. 逐折 Ljung-Box
    # ========================================================

    lb_df = residual_diagnostics(
        pred_df
    )


    print(
        "=" * 88
    )

    print(
        "阶段 5A：逐折 Ljung-Box 残差诊断"
    )

    print(
        "=" * 88
    )


    print(
        "每折只有 4 个残差，"
        "因此使用 lag=1；"
        "不同折绝不汇集。"
    )


    print(
        "p < 0.05：拒绝白噪声原假设；"
        "p >= 0.05：未发现显著 lag-1 自相关。"
    )


    print(
        "注意：n=4 非常小，因此检验功效很弱，"
        "不能把 p>=0.05 写成“证明是白噪声”。"
    )


    # --------------------------------------------------------
    # 每折分别输出
    # --------------------------------------------------------

    for fold_name in (
        lb_df[
            "fold"
        ]
        .drop_duplicates()
    ):

        print()

        print(
            fold_name
        )


        rows = lb_df[
            lb_df["fold"]
            == fold_name
        ]


        for _, r in (
            rows.iterrows()
        ):

            conclusion = (

                "检测到结构"

                if r[
                    "reject_white_noise_0.05"
                ]

                else

                "未拒绝白噪声"
            )


            print(
                f"  {r['model']:<16} "
                f"LB={r['lb_stat']:.6f} "
                f"p={r['lb_pvalue']:.6f}  "
                f"{conclusion}"
            )


    # ========================================================
    # B. 泄露反事实
    # ========================================================

    leak_df = (
        leakage_counterfactual(
            series,
            metric_df
        )
    )


    # --------------------------------------------------------
    # 只用 Fold 1~5 算 CV 均值
    # --------------------------------------------------------

    cv = leak_df[
        leak_df[
            "fold"
        ]
        != "Holdout"
    ]


    correct_cv = float(
        cv[
            "correct_MASE"
        ].mean()
    )


    leaky_cv = float(
        cv[
            "leaky_MASE"
        ].mean()
    )


    # --------------------------------------------------------
    # 表面改善百分比
    # --------------------------------------------------------

    apparent_gain = (
        (
            correct_cv
            - leaky_cv
        )
        / correct_cv
        * 100.0
    )


    hold = (
        leak_df[
            leak_df[
                "fold"
            ]
            == "Holdout"
        ]
        .iloc[0]
    )


    print()

    print(
        "=" * 88
    )

    print(
        "阶段 5B："
        "数据泄露反事实"
        "（故意错误，只用于展示危害）"
    )

    print(
        "=" * 88
    )


    print(
        "泄露方式："
        "NLinear 标准化 mean/std "
        "改用完整 168 点序列统计量。"
    )


    print(
        "alpha 仍然只用训练段内部验证选择，"
        "因此只隔离“标准化泄露”这一项。"
    )


    print()


    for _, r in (
        leak_df.iterrows()
    ):

        print(
            f"{r['fold']:<8} "

            f"correct="
            f"{r['correct_MASE']:.6f}  "

            f"leaky="
            f"{r['leaky_MASE']:.6f}  "

            f"change="
            f"{r['MASE_change_leaky_minus_correct']:+.6f}"
        )


    print()


    print(
        "NLinear 5-fold CV mean MASE: "
        f"{correct_cv:.6f} "
        "-> "
        f"{leaky_cv:.6f}"
    )


    print(
        "表面改善幅度 = "
        f"{apparent_gain:.2f}%"
        "  （这是泄露造成的虚假优势）"
    )


    print(
        "Holdout MASE: "
        f"{hold['correct_MASE']:.6f} "
        "-> "
        f"{hold['leaky_MASE']:.6f}"
    )


    # ========================================================
    # 保存 CSV
    # ========================================================

    lb_path = (
        OUT
        / "ljungbox_by_fold.csv"
    )


    leak_path = (
        OUT
        / "leakage_counterfactual.csv"
    )


    summary_path = (
        OUT
        / "leakage_summary.csv"
    )


    lb_df.to_csv(
        lb_path,
        index=False,
        encoding="utf-8-sig"
    )


    leak_df.to_csv(
        leak_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # 单独保存报告中最需要的成对数字
    # --------------------------------------------------------

    pd.DataFrame(
        [
            {
                "model":
                    "NLinear",

                "metric":
                    "5-fold CV mean MASE",

                "correct":
                    correct_cv,

                "leaky":
                    leaky_cv,

                "apparent_gain_percent":
                    apparent_gain,
            },

            {
                "model":
                    "NLinear",

                "metric":
                    "Holdout MASE",

                "correct":
                    float(
                        hold[
                            "correct_MASE"
                        ]
                    ),

                "leaky":
                    float(
                        hold[
                            "leaky_MASE"
                        ]
                    ),

                "apparent_gain_percent":
                    (
                        (
                            float(
                                hold[
                                    "correct_MASE"
                                ]
                            )
                            -
                            float(
                                hold[
                                    "leaky_MASE"
                                ]
                            )
                        )
                        /
                        float(
                            hold[
                                "correct_MASE"
                            ]
                        )
                        * 100.0
                    ),
            },
        ]
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 保存图片
    # ========================================================

    fig1 = save_ljungbox_plot(
        lb_df
    )


    fig2 = save_leakage_plot(
        leak_df
    )


    print()


    for p in [
        lb_path,
        leak_path,
        summary_path,
        fig1,
        fig2,
    ]:

        print(
            "[保存]",
            p
        )


    print()

    print(
        "✅ 阶段 5 完成。"
    )


    print(
        "下一步："
        "误差时段分析 + "
        "附加任务（Direct vs Recursive 的 1~4 步误差曲线）。"
    )


# ============================================================
# 10. 程序入口
# ============================================================

if __name__ == "__main__":
    main()