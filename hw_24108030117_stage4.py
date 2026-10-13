# -*- coding: utf-8 -*-
"""
学号：24108030117
阶段 4：NLinear + 三模型逐折比较

前提：
1) hw_24108030117_stage3.py 已经跑通；
2) student_output/series_24108030117.csv 已存在。

NLinear：
- 输入窗口减去窗口末值；
- 未来 4 步目标也减去同一个末值；
- 在相对末值空间做直接式多步 Ridge；
- 预测后加回窗口末值。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 复用阶段 3 已经验证过的函数
# ============================================================

from hw_24108030117_stage3 import (
    SID,
    LOOK_BACK,
    HORIZON,
    PERIOD,
    N_FOLDS,
    ROLLING_K,
    EXPECTED_ANCHOR,
    INPUT,
    load_series,
    rolling_origin_folds,
    naive_scale,
    metrics,
    seasonal_naive,
    ridge_predict,
    scaler,
    ridge_fit,
    select_alpha,
)


# ============================================================
# 1. 输出目录
# ============================================================

HERE = Path(__file__).resolve().parent

OUT = (
    HERE
    / "student_output"
    / "stage4"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. NLinear 特征
# ============================================================

def nlinear_feature(
    window: np.ndarray,
    target_start: pd.Timestamp
):
    """
    返回：
    1. 90 维 NLinear 因果特征
    2. 当前窗口最后一个值 base

    特征：
    60 维相对窗口
    + 6 个工程特征
    + 24 维小时 one-hot
    """

    w = np.asarray(
        window,
        dtype=float
    )


    # --------------------------------------------------------
    # 基础检查
    # --------------------------------------------------------

    if len(w) != LOOK_BACK:

        raise ValueError(
            f"窗口必须为 {LOOK_BACK} 步，"
            f"当前为 {len(w)}。"
        )


    if not np.isfinite(
        w
    ).all():

        raise ValueError(
            "窗口包含 NaN / Inf。"
        )


    # --------------------------------------------------------
    # NLinear 实例归一化基准
    #
    # base 就是窗口最后一个真实值
    # --------------------------------------------------------

    base = float(
        w[-1]
    )


    # 最近半周期
    # period = 24
    # rolling_k = 12
    recent = (
        w[-ROLLING_K:]
    )


    # --------------------------------------------------------
    # NLinear 核心
    #
    # 完整窗口全部减去最后一个值
    #
    # 相当于：
    #
    # x'_t = x_t - x_last
    #
    # 让模型学习“相对变化”
    # --------------------------------------------------------

    relative_window = (
        w - base
    )


    # --------------------------------------------------------
    # 工程特征
    #
    # 所有“位置型数值”也进入相对末值空间
    #
    # 标准差本身是波动尺度，因此不减 base
    # --------------------------------------------------------

    extra = np.array(
        [
            # lag 1
            w[-1] - base,

            # lag 2
            w[-2] - base,

            # lag 3
            w[-3] - base,

            # lag 24
            w[-PERIOD] - base,

            # 最近 12 步滚动均值
            np.mean(
                recent
            ) - base,

            # 最近 12 步滚动标准差
            np.std(
                recent,
                ddof=0
            ),
        ],
        dtype=float
    )


    # --------------------------------------------------------
    # 预测起点小时 one-hot
    #
    # 例如预测起点为 08:00
    #
    # 第 8 个位置 = 1
    # 其余 = 0
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


    # --------------------------------------------------------
    # 拼接
    #
    # 60 + 6 + 24 = 90
    # --------------------------------------------------------

    feat = np.concatenate(
        [
            relative_window,
            extra,
            hour
        ]
    )


    expected_dim = (
        LOOK_BACK
        + 6
        + 24
    )


    if feat.shape != (
        expected_dim,
    ):

        raise AssertionError(
            f"特征维度错误："
            f"{feat.shape}"
        )


    return (
        feat,
        base
    )


# ============================================================
# 3. 构造 NLinear 监督学习样本
# ============================================================

def make_nlinear_xy(
    train: pd.Series
):

    if len(train) < (
        LOOK_BACK
        + HORIZON
    ):

        raise ValueError(
            "训练段不足以构造 "
            "NLinear 样本。"
        )


    y = train.to_numpy(
        dtype=float
    )


    X = []
    Y = []


    # --------------------------------------------------------
    # 每一个监督学习样本：
    #
    # 过去 60 步
    #        ↓
    # 未来 4 步
    #
    # 但是 NLinear 在相对末值空间训练
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


        feat, base = (
            nlinear_feature(
                window,
                train.index[i]
            )
        )


        # ----------------------------------------------------
        # 目标也必须减去
        # “同一个窗口末值”
        #
        # 这一点非常重要
        # ----------------------------------------------------

        target_relative = (
            y[
                i:
                i + HORIZON
            ]
            - base
        )


        X.append(
            feat
        )

        Y.append(
            target_relative
        )


    X = np.vstack(
        X
    )

    Y = np.vstack(
        Y
    )


    return (
        X,
        Y
    )


# ============================================================
# 4. NLinear 拟合 + 预测
# ============================================================

def nlinear_predict(
    train: pd.Series,
    target_start: pd.Timestamp
):

    # --------------------------------------------------------
    # 构造当前外层训练段的监督学习样本
    # --------------------------------------------------------

    X, Y = make_nlinear_xy(
        train
    )


    # --------------------------------------------------------
    # alpha 只能在当前外层训练段内部选择
    #
    # 直接复用 Stage 3 已经验证过的
    # 尾部时间验证逻辑
    # --------------------------------------------------------

    (
        best_alpha,
        alpha_table
    ) = select_alpha(
        X,
        Y
    )


    # --------------------------------------------------------
    # alpha 确定以后
    #
    # 使用整个外层训练段重新计算标准化参数
    #
    # 测试集仍然完全不可见
    # --------------------------------------------------------

    (
        x_mu,
        x_sd,
        y_mu,
        y_sd
    ) = scaler(
        X,
        Y
    )


    Xn = (
        X - x_mu
    ) / x_sd


    Yn = (
        Y - y_mu
    ) / y_sd


    # --------------------------------------------------------
    # Ridge 闭式解
    #
    # NLinear 的“Linear”部分
    # --------------------------------------------------------

    W = ridge_fit(
        Xn,
        Yn,
        best_alpha
    )


    # --------------------------------------------------------
    # 获取当前预测起点之前的最后 60 步
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


    # --------------------------------------------------------
    # NLinear 特征
    # --------------------------------------------------------

    feat, base = (
        nlinear_feature(
            window,
            target_start
        )
    )


    x = feat.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # 使用训练段标准化参数
    # --------------------------------------------------------

    xn = (
        x - x_mu
    ) / x_sd


    # --------------------------------------------------------
    # 预测的是：
    #
    # future - base
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
    # NLinear 反归一化
    #
    # ŷ = relative prediction + base
    #
    # 加回窗口最后一个真实值
    # --------------------------------------------------------

    pred = (
        pred_relative
        .ravel()
        + base
    )


    return (
        pred,
        best_alpha,
        alpha_table
    )


# ============================================================
# 5. 跑三个模型
# ============================================================

def run_three_models(
    series: pd.Series
):

    folds = rolling_origin_folds(
        len(series)
    )


    metric_rows = []
    pred_rows = []
    alpha_rows = []


    for order, f in enumerate(
        folds,
        1
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


        ytr = train.to_numpy(
            dtype=float
        )

        yte = test.to_numpy(
            dtype=float
        )


        # ----------------------------------------------------
        # 当前折 MASE 分母
        #
        # 只能使用当前训练段
        # ----------------------------------------------------

        scale = naive_scale(
            ytr
        )


        # ====================================================
        # 模型 1：Seasonal Naive
        # ====================================================

        sn = seasonal_naive(
            ytr
        )


        sm = metrics(
            yte,
            sn,
            scale
        )


        metric_rows.append(
            {
                "fold":
                    f["name"],

                "order":
                    order,

                "model":
                    "Seasonal Naive",

                "train_n":
                    len(train),

                "alpha":
                    np.nan,

                **sm
            }
        )


        # ====================================================
        # 模型 2：Ridge-AR
        #
        # 直接复用 Stage 3
        # ====================================================

        (
            rp,
            ridge_alpha,
            ridge_table
        ) = ridge_predict(
            train,
            test.index[0]
        )


        rm = metrics(
            yte,
            rp,
            scale
        )


        metric_rows.append(
            {
                "fold":
                    f["name"],

                "order":
                    order,

                "model":
                    "Ridge-AR",

                "train_n":
                    len(train),

                "alpha":
                    ridge_alpha,

                **rm
            }
        )


        # ====================================================
        # 模型 3：NLinear
        # ====================================================

        (
            npred,
            nlinear_alpha,
            nlinear_table
        ) = nlinear_predict(
            train,
            test.index[0]
        )


        nm = metrics(
            yte,
            npred,
            scale
        )


        metric_rows.append(
            {
                "fold":
                    f["name"],

                "order":
                    order,

                "model":
                    "NLinear",

                "train_n":
                    len(train),

                "alpha":
                    nlinear_alpha,

                **nm
            }
        )


        # ====================================================
        # 保存 alpha 搜索全过程
        # ====================================================

        for (
            model_name,
            chosen,
            table
        ) in [

            (
                "Ridge-AR",
                ridge_alpha,
                ridge_table
            ),

            (
                "NLinear",
                nlinear_alpha,
                nlinear_table
            ),
        ]:


            for _, r in (
                table.iterrows()
            ):

                alpha_rows.append(
                    {
                        "fold":
                            f["name"],

                        "order":
                            order,

                        "model":
                            model_name,

                        "alpha":
                            float(
                                r["alpha"]
                            ),

                        "inner_val_MAE":
                            float(
                                r["val_MAE"]
                            ),

                        "n_inner_fit":
                            int(
                                r[
                                    "n_inner_fit"
                                ]
                            ),

                        "n_inner_val":
                            int(
                                r[
                                    "n_inner_val"
                                ]
                            ),

                        "selected":
                            bool(
                                float(
                                    r["alpha"]
                                )
                                ==
                                float(
                                    chosen
                                )
                            ),
                    }
                )


        # ====================================================
        # 保存每一个 horizon 的预测结果
        # ====================================================

        for h in range(
            HORIZON
        ):

            pred_rows.append(
                {
                    "fold":
                        f["name"],

                    "order":
                        order,

                    "horizon":
                        h + 1,

                    "timestamp":
                        test.index[h],

                    "actual":
                        float(
                            yte[h]
                        ),

                    "Seasonal_Naive":
                        float(
                            sn[h]
                        ),

                    "Ridge_AR":
                        float(
                            rp[h]
                        ),

                    "NLinear":
                        float(
                            npred[h]
                        ),
                }
            )


    return (
        folds,
        pd.DataFrame(
            metric_rows
        ),
        pd.DataFrame(
            pred_rows
        ),
        pd.DataFrame(
            alpha_rows
        )
    )


# ============================================================
# 6. CV 汇总
# ============================================================

def summary_table(
    mdf: pd.DataFrame
):

    # Holdout 不进入 CV 均值
    cv = mdf[
        mdf["fold"]
        != "Holdout"
    ]


    rows = []


    for model, g in (
        cv.groupby(
            "model"
        )
    ):

        hold = mdf[
            (
                mdf["fold"]
                == "Holdout"
            )
            &
            (
                mdf["model"]
                == model
            )
        ].iloc[0]


        rows.append(
            {
                "model":
                    model,

                "CV_MAE_mean":
                    g["MAE"].mean(),

                "CV_MAE_std":
                    g["MAE"].std(
                        ddof=0
                    ),

                "CV_RMSE_mean":
                    g["RMSE"].mean(),

                "CV_RMSE_std":
                    g["RMSE"].std(
                        ddof=0
                    ),

                "CV_MASE_mean":
                    g["MASE"].mean(),

                "CV_MASE_std":
                    g["MASE"].std(
                        ddof=0
                    ),

                "Holdout_MASE":
                    hold["MASE"],
            }
        )


    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "CV_MASE_mean"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 7. 三模型逐折预测图
# ============================================================

def save_three_model_plot(
    preds: pd.DataFrame
):

    path = (
        OUT
        / "06_three_model_fold_forecasts.png"
    )


    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            13.5,
            7.2
        )
    )


    axes = (
        axes.ravel()
    )


    for ax, name in zip(
        axes,
        preds[
            "fold"
        ].drop_duplicates()
    ):

        g = preds[
            preds["fold"]
            == name
        ]


        x = g[
            "horizon"
        ]


        ax.plot(
            x,
            g["actual"],
            marker="o",
            label="Actual"
        )


        ax.plot(
            x,
            g[
                "Seasonal_Naive"
            ],
            marker="o",
            label="Seasonal Naive"
        )


        ax.plot(
            x,
            g[
                "Ridge_AR"
            ],
            marker="o",
            label="Ridge-AR"
        )


        ax.plot(
            x,
            g[
                "NLinear"
            ],
            marker="o",
            label="NLinear"
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
        "Rolling-origin forecasts: "
        "three-model comparison",
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
        ncol=4
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
        path,
        dpi=160,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    return path


# ============================================================
# 8. 三模型 5 折 MASE 图
# ============================================================

def save_mase_plot(
    mdf: pd.DataFrame
):

    path = (
        OUT
        / "07_cv_mase_by_fold.png"
    )


    cv = mdf[
        mdf["fold"]
        != "Holdout"
    ]


    fig, ax = plt.subplots(
        figsize=(
            10.5,
            5.0
        )
    )


    for model, g in (
        cv.groupby(
            "model"
        )
    ):

        g = g.sort_values(
            "order"
        )


        ax.plot(
            g["order"],
            g["MASE"],
            marker="o",
            label=model
        )


    # MASE = 1 是一个直观参考线
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="MASE = 1"
    )


    ax.set_title(
        "MASE across rolling-origin folds"
    )


    ax.set_xlabel(
        "CV fold"
    )


    ax.set_ylabel(
        "MASE"
    )


    ax.set_xticks(
        range(
            1,
            N_FOLDS + 1
        )
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


    (
        folds,
        mdf,
        pdf,
        adf
    ) = run_three_models(
        series
    )


    # --------------------------------------------------------
    # 再保护一次 Fold 1 锚点
    # --------------------------------------------------------

    anchor = float(
        mdf[
            (
                mdf["fold"]
                == "Fold 1"
            )
            &
            (
                mdf["model"]
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
        f"锚点失配："
        f"当前 {anchor:.4f}，"
        f"任务书 {EXPECTED_ANCHOR:.4f}"
    )


    sdf = summary_table(
        mdf
    )


    print(
        "=" * 82
    )

    print(
        "阶段 4：NLinear + 三模型逐折比较"
    )

    print(
        "=" * 82
    )


    print(
        "Fold 1 Seasonal Naive MASE "
        f"= {anchor:.10f} "
        f"-> {anchor:.4f}  ✅"
    )


    print(
        "NLinear feature dimension "
        f"= {LOOK_BACK + 6 + 24}"
    )


    # --------------------------------------------------------
    # 逐折输出
    # --------------------------------------------------------

    for f in folds:

        print()

        print(
            f["name"]
        )


        rows = mdf[
            mdf["fold"]
            == f["name"]
        ]


        for _, r in (
            rows.iterrows()
        ):

            if pd.isna(
                r["alpha"]
            ):

                a = "-"

            else:

                a = (
                    f"{r['alpha']:.1f}"
                )


            print(
                f"  {r['model']:<16} "
                f"alpha={a:<5} "
                f"MAE={r['MAE']:.6f} "
                f"RMSE={r['RMSE']:.6f} "
                f"MASE={r['MASE']:.6f}"
            )


        # ----------------------------------------------------
        # 输出 Ridge 与 NLinear
        # 选中的 alpha
        # ----------------------------------------------------

        selected = adf[
            (
                adf["fold"]
                == f["name"]
            )
            &
            (
                adf["selected"]
            )
        ]


        for _, r in (
            selected.iterrows()
        ):

            print(
                f"  {r['model']:<16} "
                f"selected alpha="
                f"{r['alpha']:.1f}; "
                f"inner fit/val="
                f"{int(r['n_inner_fit'])}/"
                f"{int(r['n_inner_val'])}"
            )


    # --------------------------------------------------------
    # CV 汇总
    # --------------------------------------------------------

    print()

    print(
        "=" * 82
    )

    print(
        "CV summary "
        "(Fold 1~5 only; "
        "Holdout separate)"
    )

    print(
        "=" * 82
    )


    for _, r in (
        sdf.iterrows()
    ):

        print(
            f"{r['model']:<16} "
            f"CV MASE="
            f"{r['CV_MASE_mean']:.6f} "
            f"± "
            f"{r['CV_MASE_std']:.6f}  "

            f"CV MAE="
            f"{r['CV_MAE_mean']:.6f} "
            f"± "
            f"{r['CV_MAE_std']:.6f}  "

            f"Holdout MASE="
            f"{r['Holdout_MASE']:.6f}"
        )


    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    m_path = (
        OUT
        / "three_model_fold_metrics.csv"
    )

    p_path = (
        OUT
        / "three_model_fold_predictions.csv"
    )

    a_path = (
        OUT
        / "three_model_alpha_search.csv"
    )

    s_path = (
        OUT
        / "three_model_cv_summary.csv"
    )


    mdf.to_csv(
        m_path,
        index=False,
        encoding="utf-8-sig"
    )


    pdf.to_csv(
        p_path,
        index=False,
        encoding="utf-8-sig"
    )


    adf.to_csv(
        a_path,
        index=False,
        encoding="utf-8-sig"
    )


    sdf.to_csv(
        s_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # 保存图片
    # --------------------------------------------------------

    fig1 = save_three_model_plot(
        pdf
    )

    fig2 = save_mase_plot(
        mdf
    )


    print()


    for p in [
        m_path,
        p_path,
        a_path,
        s_path,
        fig1,
        fig2
    ]:

        print(
            "[保存]",
            p
        )


    print()

    print(
        "✅ 阶段 4 完成。"
    )

    print(
        "下一步：逐折残差 Ljung-Box "
        "+ 泄露反事实。"
    )


# ============================================================
# 10. 程序入口
# ============================================================

if __name__ == "__main__":
    main()