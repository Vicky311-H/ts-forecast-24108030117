# -*- coding: utf-8 -*-
"""
24108030117 - 阶段7：汇总最终实验结果，生成报告填表数据。

作用：
1. 读取 stage4 / stage5 / stage6 已生成的 CSV；
2. 再次校验锚点和逐折结果；
3. 汇总 MASE、MAE、RMSE；
4. 整理 Ljung-Box、泄露反事实、时段误差；
5. 整理 Direct vs Recursive 附加任务；
6. 生成 report_fill_values.txt，供填写报告使用。

注意：
本程序不重新训练模型，不修改实验结果。
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 固定信息
# ============================================================

SID = "24108030117"

EXPECTED_ANCHOR = 0.2822

FINGERPRINT = "e461b6a28e6d"

MODELS = [
    "Seasonal Naive",
    "Ridge-AR",
    "NLinear",
]

CV_FOLDS = [
    f"Fold {i}"
    for i in range(1, 6)
]

# 报告模板的残差表只有一套，
# 建议填 Ridge-AR，并在正文说明三模型均已检查。
RESIDUAL_MODEL = "Ridge-AR"


# ============================================================
# 2. 路径
# ============================================================

HERE = Path(
    __file__
).resolve().parent


ROOT = (
    HERE
    / "student_output"
)


OUT = (
    ROOT
    / "stage7"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


PATHS = {

    "metrics":
        ROOT
        / "stage4"
        / "three_model_fold_metrics.csv",

    "summary":
        ROOT
        / "stage4"
        / "three_model_cv_summary.csv",

    "lb":
        ROOT
        / "stage5"
        / "ljungbox_by_fold.csv",

    "leak":
        ROOT
        / "stage5"
        / "leakage_summary.csv",

    "time":
        ROOT
        / "stage6"
        / "error_by_time_block.csv",

    "dr":
        ROOT
        / "stage6"
        / "direct_recursive_horizon_summary.csv",
}


# ============================================================
# 3. 安全读取 CSV
# ============================================================

def read_csv(
    key,
    required_columns
):

    path = PATHS[
        key
    ]


    if not path.exists():

        raise FileNotFoundError(
            f"缺少文件：{path}"
        )


    df = pd.read_csv(
        path
    )


    missing = (
        set(
            required_columns
        )
        - set(
            df.columns
        )
    )


    if missing:

        raise ValueError(
            f"{path.name} "
            f"缺少字段："
            f"{sorted(missing)}"
        )


    return df


# ============================================================
# 4. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 读取阶段 4
    # --------------------------------------------------------

    metrics = read_csv(

        "metrics",

        [
            "fold",
            "model",
            "MAE",
            "RMSE",
            "MASE",
        ]
    )


    summary = read_csv(

        "summary",

        [
            "model",

            "CV_MAE_mean",
            "CV_MAE_std",

            "CV_RMSE_mean",
            "CV_RMSE_std",

            "CV_MASE_mean",
            "CV_MASE_std",

            "Holdout_MASE",
        ]
    )


    # --------------------------------------------------------
    # 读取阶段 5
    # --------------------------------------------------------

    lb = read_csv(

        "lb",

        [
            "fold",
            "model",
            "lb_stat",
            "lb_pvalue",
        ]
    )


    leak = read_csv(

        "leak",

        [
            "model",
            "metric",
            "correct",
            "leaky",
            "apparent_gain_percent",
        ]
    )


    # --------------------------------------------------------
    # 读取阶段 6
    # --------------------------------------------------------

    time_df = read_csv(

        "time",

        [
            "scope",
            "model",
            "time_block",
            "n",
            "MAE",
            "max_AE",
        ]
    )


    dr = read_csv(

        "dr",

        [
            "strategy",
            "horizon",
            "n",
            "MAE",
            "mean_scaled_AE",
        ]
    )


    # ========================================================
    # A. 再次校验 Fold 1 锚点
    # ========================================================

    anchor_row = metrics[
        (
            metrics["fold"]
            == "Fold 1"
        )
        &
        (
            metrics["model"]
            == "Seasonal Naive"
        )
    ]


    if len(
        anchor_row
    ) != 1:

        raise ValueError(
            "Fold 1 Seasonal Naive "
            "结果不唯一。"
        )


    anchor = float(
        anchor_row
        .iloc[
            0
        ][
            "MASE"
        ]
    )


    if (
        round(
            anchor,
            4
        )
        != EXPECTED_ANCHOR
    ):

        raise AssertionError(
            f"锚点错误："
            f"{anchor:.4f} "
            f"!= "
            f"{EXPECTED_ANCHOR:.4f}"
        )


    # ========================================================
    # B. 检查三个模型折数是否完整
    # ========================================================

    expected_folds = set(
        CV_FOLDS
        + [
            "Holdout"
        ]
    )


    for model in MODELS:

        got = set(

            metrics.loc[
                metrics[
                    "model"
                ]
                == model,

                "fold"
            ]
        )


        if (
            got
            != expected_folds
        ):

            raise ValueError(
                f"{model} "
                f"折不完整："
                f"{sorted(got)}"
            )


    # ========================================================
    # C. 重新反算 CV 汇总
    # ========================================================

    cv = metrics[
        metrics[
            "fold"
        ]
        .isin(
            CV_FOLDS
        )
    ]


    for model in MODELS:

        raw = cv[
            cv[
                "model"
            ]
            == model
        ]


        s = summary[
            summary[
                "model"
            ]
            == model
        ]


        if len(
            s
        ) != 1:

            raise ValueError(
                f"{model} "
                "summary 不唯一。"
            )


        s = s.iloc[
            0
        ]


        checks = {

            "CV_MAE_mean":
                raw[
                    "MAE"
                ].mean(),

            "CV_MAE_std":
                raw[
                    "MAE"
                ].std(
                    ddof=0
                ),

            "CV_RMSE_mean":
                raw[
                    "RMSE"
                ].mean(),

            "CV_RMSE_std":
                raw[
                    "RMSE"
                ].std(
                    ddof=0
                ),

            "CV_MASE_mean":
                raw[
                    "MASE"
                ].mean(),

            "CV_MASE_std":
                raw[
                    "MASE"
                ].std(
                    ddof=0
                ),
        }


        for (
            col,
            value
        ) in checks.items():

            if not np.isclose(
                float(
                    s[
                        col
                    ]
                ),

                value,

                atol=1e-10,

                rtol=0
            ):

                raise AssertionError(
                    f"{model} 的 "
                    f"{col} "
                    "与逐折结果不一致。"
                )


    # ========================================================
    # D. MASE 主结果表
    # ========================================================

    mase_rows = []


    for model in MODELS:

        g = (

            metrics[
                metrics[
                    "model"
                ]
                == model
            ]
            .set_index(
                "fold"
            )
        )


        s = summary[
            summary[
                "model"
            ]
            == model
        ].iloc[
            0
        ]


        row = {

            "模型":
                model,
        }


        for fold in CV_FOLDS:

            row[
                fold
            ] = float(
                g.loc[
                    fold,
                    "MASE"
                ]
            )


        row[
            "末端保留"
        ] = float(
            g.loc[
                "Holdout",
                "MASE"
            ]
        )


        row[
            "均值"
        ] = float(
            s[
                "CV_MASE_mean"
            ]
        )


        row[
            "标准差"
        ] = float(
            s[
                "CV_MASE_std"
            ]
        )


        mase_rows.append(
            row
        )


    mase = pd.DataFrame(
        mase_rows
    )


    # ========================================================
    # E. MAE / RMSE 汇总
    # ========================================================

    aux = summary[
        [
            "model",

            "CV_MAE_mean",
            "CV_MAE_std",

            "CV_RMSE_mean",
            "CV_RMSE_std",
        ]
    ].copy()


    aux.columns = [

        "模型",

        "MAE均值",
        "MAE标准差",

        "RMSE均值",
        "RMSE标准差",
    ]


    # ========================================================
    # F. Ljung-Box
    #
    # 模板只有一套残差表
    # 使用 Ridge-AR
    # ========================================================

    resid = lb[
        (
            lb[
                "model"
            ]
            == RESIDUAL_MODEL
        )
        &
        (
            lb[
                "fold"
            ]
            .isin(
                CV_FOLDS
            )
        )
    ][
        [
            "fold",
            "lb_stat",
            "lb_pvalue",
        ]
    ].copy()


    if (
        set(
            resid[
                "fold"
            ]
        )
        != set(
            CV_FOLDS
        )
    ):

        raise ValueError(
            "Ridge-AR 的五折 "
            "Ljung-Box 不完整。"
        )


    resid[
        "残差是否仍有显著lag1结构"
    ] = np.where(

        resid[
            "lb_pvalue"
        ]
        < 0.05,

        "有",

        "未检出",
    )


    order = {

        fold:
            i

        for (
            i,
            fold
        ) in enumerate(
            CV_FOLDS
        )
    }


    resid[
        "_order"
    ] = (

        resid[
            "fold"
        ]
        .map(
            order
        )
    )


    resid = (

        resid
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
    )


    # ========================================================
    # G. 泄露反事实
    # ========================================================

    leak_row = leak[
        (
            leak[
                "model"
            ]
            == "NLinear"
        )
        &
        (
            leak[
                "metric"
            ]
            == "5-fold CV mean MASE"
        )
    ]


    if len(
        leak_row
    ) != 1:

        raise ValueError(
            "找不到 NLinear 的 "
            "5-fold CV mean MASE "
            "泄露结果。"
        )


    leak_row = (
        leak_row
        .iloc[
            0
        ]
    )


    # ========================================================
    # H. 误差最大时段
    # ========================================================

    time_cv = time_df[
        time_df[
            "scope"
        ]
        == "CV"
    ]


    time_rows = []


    for model in MODELS:

        g = (

            time_cv[
                time_cv[
                    "model"
                ]
                == model
            ]
            .sort_values(
                "MAE",
                ascending=False
            )
        )


        if g.empty:

            raise ValueError(
                f"{model} "
                "缺少时段误差。"
            )


        r = g.iloc[
            0
        ]


        time_rows.append(
            {
                "模型":
                    model,

                "最大误差时段":
                    r[
                        "time_block"
                    ],

                "n":
                    int(
                        r[
                            "n"
                        ]
                    ),

                "MAE":
                    float(
                        r[
                            "MAE"
                        ]
                    ),

                "maxAE":
                    float(
                        r[
                            "max_AE"
                        ]
                    ),
            }
        )


    time_summary = pd.DataFrame(
        time_rows
    )


    # ========================================================
    # I. Direct vs Recursive
    # ========================================================

    dr_rows = []


    for strategy in [
        "Direct",
        "Recursive"
    ]:


        g = (

            dr[
                dr[
                    "strategy"
                ]
                == strategy
            ]
            .sort_values(
                "horizon"
            )
        )


        hs = set(
            g[
                "horizon"
            ]
            .astype(
                int
            )
        )


        if (
            hs
            != {
                1,
                2,
                3,
                4,
            }
        ):

            raise ValueError(
                f"{strategy} "
                f"horizon 不完整："
                f"{sorted(hs)}"
            )


        row = {

            "策略":
                strategy,
        }


        for r in (
            g.itertuples()
        ):

            row[
                f"h={int(r.horizon)} MAE"
            ] = float(
                r.MAE
            )


        row[
            "四步MAE均值"
        ] = float(
            g[
                "MAE"
            ].mean()
        )


        dr_rows.append(
            row
        )


    dr_summary = pd.DataFrame(
        dr_rows
    )


    direct_mean = float(

        dr_summary.loc[
            dr_summary[
                "策略"
            ]
            == "Direct",

            "四步MAE均值"
        ]
        .iloc[
            0
        ]
    )


    recursive_mean = float(

        dr_summary.loc[
            dr_summary[
                "策略"
            ]
            == "Recursive",

            "四步MAE均值"
        ]
        .iloc[
            0
        ]
    )


    direct_gain = (

        (
            recursive_mean
            - direct_mean
        )

        / recursive_mean

        * 100.0
    )


    # ========================================================
    # J. 保存 CSV
    # ========================================================

    outputs = {

        "final_mase_table.csv":
            mase,

        "final_mae_rmse_table.csv":
            aux,

        "final_ljungbox_ridge.csv":
            resid,

        "final_time_block_summary.csv":
            time_summary,

        "final_direct_recursive_summary.csv":
            dr_summary,
    }


    for (
        name,
        df
    ) in outputs.items():

        df.to_csv(

            OUT
            / name,

            index=False,

            encoding="utf-8-sig"
        )


    # ========================================================
    # K. 生成报告填表值
    # ========================================================

    txt = (

        OUT
        / "report_fill_values.txt"
    )


    with txt.open(
        "w",
        encoding="utf-8"
    ) as f:


        f.write(
            f"学号：{SID}\n"
        )


        f.write(
            f"数据指纹："
            f"{FINGERPRINT}\n"
        )


        f.write(
            f"锚点："
            f"{anchor:.4f}\n"
        )


        f.write(
            "\n"
            "目标变量：Gap (s)\n"
            "方向/车道：South / SB_NS\n"
            "异常区间：[0.45, 57.0]\n"
            "粒度：1h\n"
            "最后7天\n"
            "序列长度：168\n"
            "period：24\n"
            "look_back：60\n"
            "horizon：4\n"
            "CV：5折 + 末端保留\n"
            "随机种子：56426\n"
        )


        f.write(
            "\n"
            "原始事件：503768\n"
            "方向+车道筛选后：97770\n"
            "异常剔除后：93724\n"
            "最后7天事件：81836\n"
            "聚合后：168点\n"
            "季节ACF峰：lag=24，"
            "ACF=0.807923\n"
            "明显趋势：无\n"
        )


        f.write(
            "\nMASE：\n"
        )


        f.write(
            mase.to_string(
                index=False
            )
        )


        f.write(
            "\n\n"
            "MAE / RMSE：\n"
        )


        f.write(
            aux.to_string(
                index=False
            )
        )


        f.write(
            "\n\n"
            "泄露反事实：\n"
        )


        f.write(
            "NLinear 5折CV mean MASE："

            f"{float(leak_row['correct']):.6f}"

            " -> "

            f"{float(leak_row['leaky']):.6f}"

            "，表面改善 "

            f"{float(leak_row['apparent_gain_percent']):.2f}%\n"
        )


        f.write(
            "\nLjung-Box（Ridge-AR）：\n"
        )


        f.write(
            resid.to_string(
                index=False
            )
        )


        f.write(
            "\n\n误差最大时段：\n"
        )


        f.write(
            time_summary.to_string(
                index=False
            )
        )


        f.write(
            "\n\nDirect vs Recursive：\n"
        )


        f.write(
            dr_summary.to_string(
                index=False
            )
        )


        f.write(
            "\n"
            "Direct 四步平均 MAE "
            "相对 Recursive 低 "

            f"{direct_gain:.2f}%\n"
        )


    # ========================================================
    # L. 控制台输出
    # ========================================================

    print(
        "=" * 88
    )


    print(
        "阶段 7：最终汇总"
    )


    print(
        "=" * 88
    )


    print(
        f"数据指纹 = "
        f"{FINGERPRINT}"
    )


    print(
        f"锚点 = "
        f"{anchor:.4f} ✅"
    )


    print()


    print(
        "【MASE】"
    )


    print(

        mase.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}"
        )
    )


    print()


    print(
        "【MAE / RMSE】"
    )


    for r in (
        aux.itertuples(
            index=False
        )
    ):


        print(
            f"{r.模型:<16} "

            f"MAE="
            f"{r.MAE均值:.6f} "
            f"± "
            f"{r.MAE标准差:.6f}  "

            f"RMSE="
            f"{r.RMSE均值:.6f} "
            f"± "
            f"{r.RMSE标准差:.6f}"
        )


    print()


    print(
        "【泄露反事实】"
    )


    print(
        "NLinear 5折CV mean MASE: "

        f"{float(leak_row['correct']):.6f}"

        " -> "

        f"{float(leak_row['leaky']):.6f}"

        "，表面改善 "

        f"{float(leak_row['apparent_gain_percent']):.2f}%"
    )


    print()


    print(
        f"【Ljung-Box："
        f"{RESIDUAL_MODEL}】"
    )


    print(

        resid.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}"
        )
    )


    print()


    print(
        "【误差最大时段】"
    )


    print(

        time_summary.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}"
        )
    )


    print()


    print(
        "【Direct vs Recursive】"
    )


    print(

        dr_summary.to_string(

            index=False,

            float_format=lambda x:
                f"{x:.6f}"
        )
    )


    print(
        f"Direct 相对 Recursive "
        f"的四步平均 MAE 低 "
        f"{direct_gain:.2f}%"
    )


    print()


    print(
        "保存到：",
        OUT
    )


    print(
        "✅ 阶段 7 完成。"
    )


if __name__ == "__main__":
    main()