# -*- coding: utf-8 -*-
"""
学号：24108030117
阶段 6：误差时段分析 + 附加任务 Direct vs Recursive (h=1~4)

原则：
1. 误差时段分析以 Fold 1~5 的 out-of-sample 预测为主，
   Holdout 单独保留，不参与主分析。

2. Direct：
   为 h=1,2,3,4 分别训练一个 Ridge 模型，
   直接预测对应步长。

3. Recursive：
   只训练一个 1-step Ridge 模型，
   预测值逐步回填到窗口中，再预测下一步。

4. Direct / Recursive 的 alpha 都只在当前外层训练段内部
   做时间顺序尾部验证。

5. 所有标准化统计量都只来自当前训练段，
   绝不使用外层测试集。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 复用阶段 3 已经验证过的函数
# ============================================================

from hw_24108030117_stage3 import (
    LOOK_BACK,
    HORIZON,
    N_FOLDS,
    INPUT,
    load_series,
    rolling_origin_folds,
    naive_scale,
    feature,
    scaler,
    ridge_fit,
    select_alpha,
)


# ============================================================
# 复用阶段 4 的三模型结果
# ============================================================

from hw_24108030117_stage4 import (
    run_three_models,
)


# ============================================================
# 1. 输出目录
# ============================================================

HERE = Path(__file__).resolve().parent

OUT = (
    HERE
    / "student_output"
    / "stage6"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 模型列
# ============================================================

MODEL_COLUMNS = {

    "Seasonal Naive":
        "Seasonal_Naive",

    "Ridge-AR":
        "Ridge_AR",

    "NLinear":
        "NLinear",
}


# ============================================================
# 3. 时段定义
# ============================================================

TIME_BLOCKS = [

    (
        "Night (00-05)",
        0,
        5
    ),

    (
        "Morning (06-11)",
        6,
        11
    ),

    (
        "Afternoon (12-17)",
        12,
        17
    ),

    (
        "Evening (18-23)",
        18,
        23
    ),
]


# ============================================================
# 4. 根据小时判断所属时段
# ============================================================

def time_block(
    hour: int
) -> str:

    h = int(
        hour
    )

    for (
        name,
        lo,
        hi
    ) in TIME_BLOCKS:

        if (
            lo
            <= h
            <= hi
        ):

            return name


    raise ValueError(
        f"非法小时：{hour}"
    )


# ============================================================
# 5. 误差时段分析
# ============================================================

def error_time_analysis(
    predictions: pd.DataFrame
):

    """
    主分析只使用 Fold 1~5。

    Holdout 单独保留，
    不参与主要时段结论。
    """

    df = predictions.copy()


    # --------------------------------------------------------
    # 时间字段
    # --------------------------------------------------------

    df[
        "timestamp"
    ] = pd.to_datetime(
        df[
            "timestamp"
        ]
    )


    df[
        "hour"
    ] = (
        df[
            "timestamp"
        ]
        .dt.hour
        .astype(
            int
        )
    )


    df[
        "time_block"
    ] = (
        df[
            "hour"
        ]
        .map(
            time_block
        )
    )


    # --------------------------------------------------------
    # 按模型 / 时段计算绝对误差
    # --------------------------------------------------------

    rows = []


    for (
        model,
        col
    ) in MODEL_COLUMNS.items():


        error_col = (
            f"abs_error__{col}"
        )


        df[
            error_col
        ] = np.abs(
            df[
                "actual"
            ]
            - df[
                col
            ]
        )


        # ----------------------------------------------------
        # CV 与 Holdout 分开
        # ----------------------------------------------------

        for (
            scope,
            mask
        ) in [

            (
                "CV",
                df[
                    "fold"
                ]
                != "Holdout"
            ),

            (
                "Holdout",
                df[
                    "fold"
                ]
                == "Holdout"
            ),
        ]:


            sub = (
                df
                .loc[
                    mask
                ]
                .copy()
            )


            for (
                block_name,
                _,
                _
            ) in TIME_BLOCKS:


                b = sub[
                    sub[
                        "time_block"
                    ]
                    == block_name
                ]


                if len(
                    b
                ) == 0:

                    continue


                rows.append(
                    {
                        "scope":
                            scope,

                        "model":
                            model,

                        "time_block":
                            block_name,

                        "n":
                            int(
                                len(
                                    b
                                )
                            ),

                        "MAE":
                            float(
                                b[
                                    error_col
                                ]
                                .mean()
                            ),

                        "max_AE":
                            float(
                                b[
                                    error_col
                                ]
                                .max()
                            ),
                    }
                )


    block_df = pd.DataFrame(
        rows
    )


    # ========================================================
    # 查看 CV 实际覆盖了哪些小时
    # ========================================================

    cv = df[
        df[
            "fold"
        ]
        != "Holdout"
    ]


    coverage = (
        cv
        .groupby(
            "hour",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size":
                    "n_test_points"
            }
        )
    )


    # ========================================================
    # 同时保存逐小时误差
    # ========================================================

    hour_rows = []


    for (
        model,
        col
    ) in MODEL_COLUMNS.items():


        for (
            hour,
            g
        ) in cv.groupby(
            "hour"
        ):


            hour_rows.append(
                {
                    "model":
                        model,

                    "hour":
                        int(
                            hour
                        ),

                    "n":
                        int(
                            len(
                                g
                            )
                        ),

                    "MAE":
                        float(
                            np.mean(
                                np.abs(
                                    g[
                                        "actual"
                                    ]
                                    - g[
                                        col
                                    ]
                                )
                            )
                        ),
                }
            )


    hour_df = pd.DataFrame(
        hour_rows
    )


    return (
        block_df,
        hour_df,
        coverage
    )


# ============================================================
# 6. 原始 Gap 时段统计
# ============================================================

def gap_time_stats(
    series: pd.Series
) -> pd.DataFrame:

    """
    对完整 168 点序列按时段统计 Gap(s)。

    用来解释：
    为什么某些时段预测误差更大。
    """

    df = (
        series
        .rename(
            "Gap (s)"
        )
        .to_frame()
    )


    df[
        "hour"
    ] = (
        df
        .index
        .hour
        .astype(
            int
        )
    )


    df[
        "time_block"
    ] = (
        df[
            "hour"
        ]
        .map(
            time_block
        )
    )


    out = (
        df
        .groupby(
            "time_block",
            observed=False
        )[
            "Gap (s)"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # 保持时间顺序
    # --------------------------------------------------------

    order = {

        name:
            i

        for i, (
            name,
            _,
            _
        ) in enumerate(
            TIME_BLOCKS
        )
    }


    out[
        "_order"
    ] = (
        out[
            "time_block"
        ]
        .map(
            order
        )
    )


    out = (
        out
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )


    return out


# ============================================================
# 7. Direct：构造第 h 步监督学习样本
# ============================================================

def make_direct_xy(
    train: pd.Series,
    horizon_step: int
):

    """
    Direct 策略：

    h=1：
        过去 60 步 -> t+1

    h=2：
        过去 60 步 -> t+2

    h=3：
        过去 60 步 -> t+3

    h=4：
        过去 60 步 -> t+4

    四个 horizon 分别训练独立模型。

    为了公平比较，
    四个模型使用相同数量的窗口。
    """

    h = int(
        horizon_step
    )


    if (
        h < 1
        or h > HORIZON
    ):

        raise ValueError(
            f"horizon_step 必须在 "
            f"1..{HORIZON}，"
            f"当前为 {h}"
        )


    y = (
        train
        .to_numpy(
            dtype=float
        )
    )


    if len(
        y
    ) < (
        LOOK_BACK
        + HORIZON
    ):

        raise ValueError(
            "训练段太短，"
            "无法构造 Direct 样本。"
        )


    X = []
    Y = []


    # --------------------------------------------------------
    # i 表示预测起点
    #
    # 所有 Direct 模型都要求未来 4 步完整存在，
    # 从而保证训练窗口数量一致。
    # --------------------------------------------------------

    for i in range(
        LOOK_BACK,
        len(
            y
        )
        - HORIZON
        + 1
    ):


        window = (
            y[
                i
                - LOOK_BACK:
                i
            ]
        )


        target_pos = (
            i
            + h
            - 1
        )


        # ----------------------------------------------------
        # target 的小时属于已知日历信息，
        # 可以安全作为特征使用。
        # ----------------------------------------------------

        X.append(
            feature(
                window,
                train.index[
                    target_pos
                ]
            )
        )


        Y.append(
            [
                y[
                    target_pos
                ]
            ]
        )


    return (
        np.vstack(
            X
        ),
        np.vstack(
            Y
        )
    )


# ============================================================
# 8. Direct：预测某一个 horizon
# ============================================================

def fit_predict_direct_one_h(
    train: pd.Series,
    test_index: pd.DatetimeIndex,
    horizon_step: int
):

    # --------------------------------------------------------
    # 构造训练样本
    # --------------------------------------------------------

    X, Y = make_direct_xy(
        train,
        horizon_step
    )


    # --------------------------------------------------------
    # alpha 只能在当前训练段内部选择
    # --------------------------------------------------------

    (
        best_alpha,
        alpha_table
    ) = select_alpha(
        X,
        Y
    )


    # --------------------------------------------------------
    # 正确标准化
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
        X
        - x_mu
    ) / x_sd


    Yn = (
        Y
        - y_mu
    ) / y_sd


    # --------------------------------------------------------
    # Ridge 拟合
    # --------------------------------------------------------

    W = ridge_fit(
        Xn,
        Yn,
        best_alpha
    )


    # --------------------------------------------------------
    # 最后 60 步历史
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
    # 第 h 步实际目标时间
    # --------------------------------------------------------

    target_time = pd.Timestamp(
        test_index[
            horizon_step
            - 1
        ]
    )


    x = (
        feature(
            window,
            target_time
        )
        .reshape(
            1,
            -1
        )
    )


    # --------------------------------------------------------
    # 预测
    # --------------------------------------------------------

    pred = (
        (
            (
                (
                    x
                    - x_mu
                )
                / x_sd
            )
            @ W
        )
        * y_sd
        + y_mu
    )


    return (
        float(
            pred
            .ravel()[
                0
            ]
        ),

        float(
            best_alpha
        ),

        alpha_table
    )


# ============================================================
# 9. Recursive：一步模型训练样本
# ============================================================

def make_recursive_xy(
    train: pd.Series
):

    """
    Recursive：

    只训练一个一步预测模型：

    过去 60 步 -> 下一步
    """

    y = (
        train
        .to_numpy(
            dtype=float
        )
    )


    if len(
        y
    ) < (
        LOOK_BACK
        + 1
    ):

        raise ValueError(
            "训练段太短，"
            "无法构造 Recursive 样本。"
        )


    X = []
    Y = []


    for i in range(
        LOOK_BACK,
        len(
            y
        )
    ):


        X.append(
            feature(
                y[
                    i
                    - LOOK_BACK:
                    i
                ],

                train.index[
                    i
                ]
            )
        )


        Y.append(
            [
                y[
                    i
                ]
            ]
        )


    return (
        np.vstack(
            X
        ),
        np.vstack(
            Y
        )
    )


# ============================================================
# 10. Recursive：递归预测 4 步
# ============================================================

def fit_predict_recursive(
    train: pd.Series,
    test_index: pd.DatetimeIndex
):

    # --------------------------------------------------------
    # 一步训练数据
    # --------------------------------------------------------

    X, Y = make_recursive_xy(
        train
    )


    # --------------------------------------------------------
    # 内层验证选择 alpha
    # --------------------------------------------------------

    (
        best_alpha,
        alpha_table
    ) = select_alpha(
        X,
        Y
    )


    # --------------------------------------------------------
    # 标准化参数只来自训练段
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


    W = ridge_fit(
        (
            X
            - x_mu
        )
        / x_sd,

        (
            Y
            - y_mu
        )
        / y_sd,

        best_alpha
    )


    # --------------------------------------------------------
    # 只保留最后 60 个真实历史点
    # --------------------------------------------------------

    history = (
        train
        .iloc[
            -LOOK_BACK:
        ]
        .to_numpy(
            dtype=float
        )
        .tolist()
    )


    preds = []


    # --------------------------------------------------------
    # 连续预测 t+1...t+4
    # --------------------------------------------------------

    for step in range(
        HORIZON
    ):


        window = np.asarray(
            history[
                -LOOK_BACK:
            ],
            dtype=float
        )


        target_time = pd.Timestamp(
            test_index[
                step
            ]
        )


        x = (
            feature(
                window,
                target_time
            )
            .reshape(
                1,
                -1
            )
        )


        pred = float(
            (
                (
                    (
                        (
                            x
                            - x_mu
                        )
                        / x_sd
                    )
                    @ W
                )
                * y_sd
                + y_mu
            )
            .ravel()[
                0
            ]
        )


        if not np.isfinite(
            pred
        ):

            raise ValueError(
                "Recursive 产生了非有限预测。"
            )


        preds.append(
            pred
        )


        # ----------------------------------------------------
        # Recursive 的关键：
        #
        # 下一步输入使用当前“预测值”，
        # 而不是偷看真实测试值。
        # ----------------------------------------------------

        history.append(
            pred
        )


    return (
        np.asarray(
            preds,
            dtype=float
        ),

        float(
            best_alpha
        ),

        alpha_table
    )


# ============================================================
# 11. Direct vs Recursive 实验
# ============================================================

def direct_recursive_experiment(
    series: pd.Series
):

    folds = rolling_origin_folds(
        len(
            series
        )
    )


    pred_rows = []
    alpha_rows = []


    for (
        order,
        f
    ) in enumerate(
        folds,
        start=1
    ):


        tr0, tr1 = (
            f[
                "train"
            ]
        )


        te0, te1 = (
            f[
                "test"
            ]
        )


        train = (
            series
            .iloc[
                tr0:
                tr1
            ]
        )


        test = (
            series
            .iloc[
                te0:
                te1
            ]
        )


        actual = (
            test
            .to_numpy(
                dtype=float
            )
        )


        # ----------------------------------------------------
        # 当前折 MASE scale
        # ----------------------------------------------------

        scale = naive_scale(
            train
            .to_numpy(
                dtype=float
            )
        )


        # ====================================================
        # A. Direct
        # ====================================================

        direct_preds = []


        for h in range(
            1,
            HORIZON + 1
        ):


            (
                pred,
                alpha,
                table
            ) = fit_predict_direct_one_h(
                train,
                test.index,
                h
            )


            direct_preds.append(
                pred
            )


            alpha_rows.append(
                {
                    "fold":
                        f[
                            "name"
                        ],

                    "order":
                        order,

                    "strategy":
                        "Direct",

                    "horizon":
                        h,

                    "selected_alpha":
                        alpha,

                    "inner_best_MAE":
                        float(
                            table
                            .iloc[
                                0
                            ][
                                "val_MAE"
                            ]
                        ),

                    "n_inner_fit":
                        int(
                            table
                            .iloc[
                                0
                            ][
                                "n_inner_fit"
                            ]
                        ),

                    "n_inner_val":
                        int(
                            table
                            .iloc[
                                0
                            ][
                                "n_inner_val"
                            ]
                        ),
                }
            )


        # ====================================================
        # B. Recursive
        # ====================================================

        (
            recursive_preds,
            rec_alpha,
            rec_table
        ) = fit_predict_recursive(
            train,
            test.index
        )


        alpha_rows.append(
            {
                "fold":
                    f[
                        "name"
                    ],

                "order":
                    order,

                "strategy":
                    "Recursive",

                # Recursive 只有一个一步模型
                "horizon":
                    0,

                "selected_alpha":
                    rec_alpha,

                "inner_best_MAE":
                    float(
                        rec_table
                        .iloc[
                            0
                        ][
                            "val_MAE"
                        ]
                    ),

                "n_inner_fit":
                    int(
                        rec_table
                        .iloc[
                            0
                        ][
                            "n_inner_fit"
                        ]
                    ),

                "n_inner_val":
                    int(
                        rec_table
                        .iloc[
                            0
                        ][
                            "n_inner_val"
                        ]
                    ),
            }
        )


        # ====================================================
        # 保存每一个 horizon 的预测误差
        # ====================================================

        for h in range(
            HORIZON
        ):


            for (
                strategy,
                pred
            ) in [

                (
                    "Direct",
                    float(
                        direct_preds[
                            h
                        ]
                    )
                ),

                (
                    "Recursive",
                    float(
                        recursive_preds[
                            h
                        ]
                    )
                ),
            ]:


                ae = abs(
                    float(
                        actual[
                            h
                        ]
                    )
                    - pred
                )


                pred_rows.append(
                    {
                        "fold":
                            f[
                                "name"
                            ],

                        "order":
                            order,

                        "strategy":
                            strategy,

                        "horizon":
                            h + 1,

                        "timestamp":
                            test
                            .index[
                                h
                            ],

                        "actual":
                            float(
                                actual[
                                    h
                                ]
                            ),

                        "prediction":
                            pred,

                        "absolute_error":
                            ae,

                        "scaled_absolute_error":
                            ae
                            / scale,
                    }
                )


    pred_df = pd.DataFrame(
        pred_rows
    )


    alpha_df = pd.DataFrame(
        alpha_rows
    )


    # ========================================================
    # 主曲线仅使用 Fold 1~5
    # ========================================================

    cv = pred_df[
        pred_df[
            "fold"
        ]
        != "Holdout"
    ]


    horizon_df = (
        cv
        .groupby(
            [
                "strategy",
                "horizon"
            ],
            as_index=False
        )
        .agg(
            n=(
                "absolute_error",
                "size"
            ),

            MAE=(
                "absolute_error",
                "mean"
            ),

            mean_scaled_AE=(
                "scaled_absolute_error",
                "mean"
            ),
        )
    )


    # --------------------------------------------------------
    # Holdout 单独保存
    # --------------------------------------------------------

    holdout_df = (
        pred_df[
            pred_df[
                "fold"
            ]
            == "Holdout"
        ]
        .copy()
    )


    return (
        pred_df,
        horizon_df,
        holdout_df,
        alpha_df
    )


# ============================================================
# 12. 图：时段误差
# ============================================================

def save_time_block_plot(
    block_df: pd.DataFrame
) -> Path:

    path = (
        OUT
        / "10_cv_error_by_time_block.png"
    )


    cv = (
        block_df[
            block_df[
                "scope"
            ]
            == "CV"
        ]
        .copy()
    )


    block_order = [
        x[
            0
        ]
        for x in TIME_BLOCKS
    ]


    x = np.arange(
        len(
            block_order
        ),
        dtype=float
    )


    width = 0.24


    fig, ax = plt.subplots(
        figsize=(
            11.5,
            5.2
        )
    )


    for (
        j,
        model
    ) in enumerate(
        MODEL_COLUMNS
    ):


        g = (
            cv[
                cv[
                    "model"
                ]
                == model
            ]
            .set_index(
                "time_block"
            )
        )


        vals = [

            float(
                g.loc[
                    b,
                    "MAE"
                ]
            )

            if b in g.index

            else np.nan

            for b in block_order
        ]


        ax.bar(
            x
            + (
                j
                - 1
            )
            * width,

            vals,

            width=width,

            label=model
        )


    ax.set_title(
        "CV absolute error by time block"
    )


    ax.set_xlabel(
        "Time block"
    )


    ax.set_ylabel(
        "MAE (s)"
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        block_order,
        rotation=12
    )


    ax.grid(
        axis="y",
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
# 13. 图：Direct vs Recursive horizon 曲线
# ============================================================

def save_horizon_plot(
    horizon_df: pd.DataFrame
) -> Path:

    path = (
        OUT
        / "11_direct_vs_recursive_horizon_error.png"
    )


    fig, ax = plt.subplots(
        figsize=(
            9.5,
            5.0
        )
    )


    for strategy in [
        "Direct",
        "Recursive"
    ]:


        g = (
            horizon_df[
                horizon_df[
                    "strategy"
                ]
                == strategy
            ]
            .sort_values(
                "horizon"
            )
        )


        ax.plot(
            g[
                "horizon"
            ],

            g[
                "MAE"
            ],

            marker="o",
            linewidth=1.8,

            label=strategy
        )


    ax.set_title(
        "Direct vs Recursive: error by forecast horizon"
    )


    ax.set_xlabel(
        "Forecast horizon (hour)"
    )


    ax.set_ylabel(
        "Mean absolute error (s)"
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
# 14. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 读取已经通过指纹校验的 168 点序列
    # --------------------------------------------------------

    series = load_series(
        INPUT
    )


    # --------------------------------------------------------
    # 获得三个模型的 OOS 预测
    # --------------------------------------------------------

    (
        _,
        _,
        three_model_preds,
        _
    ) = run_three_models(
        series
    )


    # ========================================================
    # A. 误差时段分析
    # ========================================================

    print(
        "=" * 88
    )


    print(
        "阶段 6A："
        "误差时段分析（主分析仅 Fold 1~5）"
    )


    print(
        "=" * 88
    )


    (
        block_df,
        hour_df,
        coverage
    ) = error_time_analysis(
        three_model_preds
    )


    gap_stats_df = gap_time_stats(
        series
    )


    # --------------------------------------------------------
    # 打印 CV 小时覆盖情况
    # --------------------------------------------------------

    print(
        "CV 测试点覆盖小时："
    )


    print(
        ", ".join(

            f"{int(r.hour):02d}:00"
            f"(n={int(r.n_test_points)})"

            for r in coverage.itertuples()
        )
    )


    print(
        "说明：CV 并未覆盖 12:00~15:00，"
        "因此不能声称已完整覆盖全天 24 小时。"
    )


    # --------------------------------------------------------
    # 全序列交通状态统计
    # --------------------------------------------------------

    print()

    print(
        "全 168 点 Gap(s) 的时段统计"
        "（用于解释交通机制）："
    )


    for r in (
        gap_stats_df
        .itertuples()
    ):


        print(
            f"  {r.time_block:<20} "
            f"n={int(r.count):<2d} "
            f"mean={r.mean:.6f} "
            f"std={r.std:.6f}"
        )


    # --------------------------------------------------------
    # 三模型时段误差
    # --------------------------------------------------------

    cv_blocks = block_df[
        block_df[
            "scope"
        ]
        == "CV"
    ]


    for model in MODEL_COLUMNS:


        g = (
            cv_blocks[
                cv_blocks[
                    "model"
                ]
                == model
            ]
            .sort_values(
                "MAE",
                ascending=False
            )
        )


        print()

        print(
            model
        )


        for r in (
            g
            .itertuples()
        ):


            print(
                f"  {r.time_block:<20} "
                f"n={r.n:<2d} "
                f"MAE={r.MAE:.6f}  "
                f"maxAE={r.max_AE:.6f}"
            )


        worst = (
            g
            .iloc[
                0
            ]
        )


        print(
            "  -> CV 中误差最大的时段："
            f"{worst['time_block']} "
            f"(MAE={worst['MAE']:.6f})"
        )


    # ========================================================
    # B. Direct vs Recursive
    # ========================================================

    print()

    print(
        "=" * 88
    )


    print(
        "阶段 6B："
        "附加任务 Direct vs Recursive（1~4 步）"
    )


    print(
        "=" * 88
    )


    (
        dr_pred,
        horizon_df,
        holdout_df,
        alpha_df
    ) = direct_recursive_experiment(
        series
    )


    # --------------------------------------------------------
    # 输出 1~4 步误差
    # --------------------------------------------------------

    for strategy in [
        "Direct",
        "Recursive"
    ]:


        print()

        print(
            strategy
        )


        g = (
            horizon_df[
                horizon_df[
                    "strategy"
                ]
                == strategy
            ]
            .sort_values(
                "horizon"
            )
        )


        for r in (
            g
            .itertuples()
        ):


            print(
                f"  h={int(r.horizon)}  "
                f"n={int(r.n)}  "
                f"MAE={r.MAE:.6f}  "
                f"mean_scaled_AE="
                f"{r.mean_scaled_AE:.6f}"
            )


    # --------------------------------------------------------
    # 四个 horizon 的紧凑平均
    # --------------------------------------------------------

    means = (
        horizon_df
        .groupby(
            "strategy"
        )[
            "MAE"
        ]
        .mean()
    )


    print()

    print(
        "Across-horizon mean of CV horizon-wise MAE:"
    )


    for strategy in [
        "Direct",
        "Recursive"
    ]:


        print(
            f"  {strategy:<10} "
            f"{means[strategy]:.6f}"
        )


    # --------------------------------------------------------
    # Holdout 单独输出
    # --------------------------------------------------------

    print()

    print(
        "Holdout "
        "（只展示，不进入上面的 CV 曲线均值）："
    )


    for strategy in [
        "Direct",
        "Recursive"
    ]:


        g = (
            holdout_df[
                holdout_df[
                    "strategy"
                ]
                == strategy
            ]
            .sort_values(
                "horizon"
            )
        )


        print(
            f"  {strategy:<10} "
            f"MAE="
            f"{g['absolute_error'].mean():.6f}"
        )


    # ========================================================
    # C. 保存 CSV
    # ========================================================

    paths = {

        "block":
            OUT
            / "error_by_time_block.csv",

        "hour":
            OUT
            / "error_by_hour_cv.csv",

        "coverage":
            OUT
            / "cv_hour_coverage.csv",

        "gap_stats":
            OUT
            / "gap_by_time_block.csv",

        "dr_pred":
            OUT
            / "direct_recursive_predictions.csv",

        "horizon":
            OUT
            / "direct_recursive_horizon_summary.csv",

        "holdout":
            OUT
            / "direct_recursive_holdout.csv",

        "alpha":
            OUT
            / "direct_recursive_alpha.csv",
    }


    block_df.to_csv(
        paths[
            "block"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    hour_df.to_csv(
        paths[
            "hour"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    coverage.to_csv(
        paths[
            "coverage"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    gap_stats_df.to_csv(
        paths[
            "gap_stats"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    dr_pred.to_csv(
        paths[
            "dr_pred"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    horizon_df.to_csv(
        paths[
            "horizon"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    holdout_df.to_csv(
        paths[
            "holdout"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    alpha_df.to_csv(
        paths[
            "alpha"
        ],
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # D. 保存图
    # ========================================================

    fig1 = save_time_block_plot(
        block_df
    )


    fig2 = save_horizon_plot(
        horizon_df
    )


    print()

    print(
        "保存结果："
    )


    for p in [
        *paths.values(),
        fig1,
        fig2,
    ]:

        print(
            "[保存]",
            p
        )


    print()

    print(
        "✅ 阶段 6 完成。"
    )


    print(
        "下一步："
        "整理三个关键决定 + "
        "汇总最终实验表 + "
        "填写报告。"
    )


# ============================================================
# 15. 程序入口
# ============================================================

if __name__ == "__main__":
    main()