# -*- coding: utf-8 -*-
"""
学号：24108030117
时间序列实验 —— 阶段 1：数据构造与数据指纹校验

个人任务参数：
目标变量：Gap (s)
方向：South
车道：SB_NS
异常区间：[0.45, 57.0]
聚合粒度：1h
使用最后 7 个有效日
目标序列长度：168
数据指纹：e461b6a28e6d

注意：
日期解析必须和老师的 hw_task_selfservice.py 保持完全一致：
pd.to_datetime(..., errors="coerce")

不要自行修改成 format="mixed"。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 个人任务参数
# ============================================================

SID = "24108030117"

# 预测目标
TARGET_COL = "Gap (s)"

# 方向：2 = South
DIRECTION = 2

# 指定车道
LANE_NAME = "SB_NS"

# 异常值范围
CLIP_LO = 0.45
CLIP_HI = 57.0

# 使用最后 7 个有效日期
N_DAYS = 7

# 时间聚合粒度
RESAMPLE_RULE = "1h"

# 老师任务书给出的检查值
EXPECTED_N_POINTS = 168
EXPECTED_FINGERPRINT = "e461b6a28e6d"


# ============================================================
# 2. 文件路径
# ============================================================

# 当前 py 文件所在目录
HERE = Path(__file__).resolve().parent

# flow.csv 与本程序放在同一个 student_package 目录
CSV_PATH = HERE / "flow.csv"


# ============================================================
# 3. 数据指纹函数
# ============================================================

def fingerprint(values: np.ndarray) -> str:
    """
    按老师任务生成器的规则计算数据指纹：

    1. 转为 float 数组
    2. 四舍五入到 6 位小数
    3. 转为二进制
    4. SHA256
    5. 取前 12 位
    """

    values = np.asarray(values, dtype=float)

    rounded = np.round(values, 6)

    sha = hashlib.sha256(
        rounded.tobytes()
    ).hexdigest()

    return sha[:12]


# ============================================================
# 4. 按个人任务书构造时间序列
# ============================================================

def build_series(csv_path: Path):

    # --------------------------------------------------------
    # 第一步：读取原始 CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    stats = {}

    stats["raw_rows"] = len(df)

    print("=" * 70)
    print("① 读取原始数据")
    print("=" * 70)

    print("原始数据行数：", len(df))
    print("字段：")
    print(df.columns.tolist())


    # --------------------------------------------------------
    # 第二步：Date 转成 datetime
    #
    # 非常重要：
    # 必须与 hw_task_selfservice.py 完全一样
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    bad_date_rows = df["Date"].isna().sum()

    stats["bad_date_rows"] = int(bad_date_rows)

    print()
    print("=" * 70)
    print("② Date 时间解析")
    print("=" * 70)

    print("时间解析失败数量：", bad_date_rows)


    # --------------------------------------------------------
    # 第三步：删除 Date 无法解析的记录
    # --------------------------------------------------------

    df = df.dropna(
        subset=["Date"]
    )

    stats["after_date_rows"] = len(df)

    print("删除无效时间后的记录数：", len(df))


    # --------------------------------------------------------
    # 第四步：筛选 South
    #
    # 数据中：
    # Direction = 1 -> North
    # Direction = 2 -> South
    # --------------------------------------------------------

    df = df[
        df["Direction"] == DIRECTION
    ]

    stats["after_direction_rows"] = len(df)

    print()
    print("=" * 70)
    print("③ 筛选方向 South")
    print("=" * 70)

    print("South 筛选后记录数：", len(df))


    # --------------------------------------------------------
    # 第五步：筛选 SB_NS 车道
    # --------------------------------------------------------

    df = df[
        df["Lane Name"] == LANE_NAME
    ]

    stats["after_lane_rows"] = len(df)

    print()
    print("=" * 70)
    print("④ 筛选车道 SB_NS")
    print("=" * 70)

    print("SB_NS 筛选后记录数：", len(df))


    # --------------------------------------------------------
    # 第六步：Gap(s) 异常值处理
    #
    # 保留：
    # 0.45 <= Gap <= 57.0
    #
    # pandas between 默认包含左右边界
    # --------------------------------------------------------

    values = df[TARGET_COL].astype(float)

    valid_mask = values.between(
        CLIP_LO,
        CLIP_HI
    )

    df = df.loc[valid_mask]

    stats["after_clip_rows"] = len(df)

    print()
    print("=" * 70)
    print("⑤ Gap(s) 异常值筛选")
    print("=" * 70)

    print(
        f"保留范围：{CLIP_LO} <= Gap(s) <= {CLIP_HI}"
    )

    print(
        "异常值筛选后记录数：",
        len(df)
    )


    # --------------------------------------------------------
    # 第七步：获取所有有效日期
    # --------------------------------------------------------

    all_days = sorted(
        df["Date"]
        .dt.date
        .unique()
    )

    stats["all_valid_days"] = all_days

    print()
    print("=" * 70)
    print("⑥ 检查有效日期")
    print("=" * 70)

    print(
        "有效日期总数：",
        len(all_days)
    )

    print(
        "全部有效日期："
    )

    for day in all_days:
        print(" ", day)


    # --------------------------------------------------------
    # 第八步：仅取最后 7 个有效日期
    # --------------------------------------------------------

    if N_DAYS < len(all_days):

        keep_days = all_days[-N_DAYS:]

        keep_set = set(
            keep_days
        )

        df = df[
            df["Date"]
            .dt.date
            .isin(keep_set)
        ]

    else:

        keep_days = all_days


    stats["keep_days"] = keep_days
    stats["after_days_rows"] = len(df)

    print()
    print("=" * 70)
    print("⑦ 取最后 7 个有效日期")
    print("=" * 70)

    print(
        "保留日期："
    )

    for day in keep_days:
        print(" ", day)

    print(
        "最后 7 天原始记录数：",
        len(df)
    )


    # --------------------------------------------------------
    # 第九步：按照 1 小时聚合
    #
    # 同一小时内所有 Gap(s) 求平均
    # --------------------------------------------------------

    series = (
        df
        .set_index("Date")[TARGET_COL]
        .resample(RESAMPLE_RULE)
        .mean()
        .dropna()
        .astype(float)
    )

    stats["series_points"] = len(series)

    print()
    print("=" * 70)
    print("⑧ 按 1 小时聚合")
    print("=" * 70)

    print(
        "聚合后序列长度：",
        len(series)
    )

    print(
        "序列起始时间：",
        series.index.min()
    )

    print(
        "序列结束时间：",
        series.index.max()
    )


    return series, stats


# ============================================================
# 5. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 检查 flow.csv 是否存在
    # --------------------------------------------------------

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"\n找不到文件：{CSV_PATH}\n"
            "请确认 hw_24108030117.py 与 flow.csv "
            "位于同一个 student_package 目录。"
        )


    # --------------------------------------------------------
    # 构造时间序列
    # --------------------------------------------------------

    series, stats = build_series(
        CSV_PATH
    )


    # --------------------------------------------------------
    # 计算数据指纹
    # --------------------------------------------------------

    values = series.to_numpy(
        dtype=float
    )

    fp = fingerprint(
        values
    )


    # --------------------------------------------------------
    # 显示最终结果
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("⑨ 正式检查点 1")
    print("=" * 70)

    print(
        "最终序列长度：",
        len(series)
    )

    print(
        "计算得到的数据指纹：",
        fp
    )

    print(
        "任务书要求的序列长度：",
        EXPECTED_N_POINTS
    )

    print(
        "任务书要求的数据指纹：",
        EXPECTED_FINGERPRINT
    )


    # --------------------------------------------------------
    # 自动进行硬性校验
    # --------------------------------------------------------

    assert len(series) == EXPECTED_N_POINTS, (
        f"\n❌ 序列长度错误！"
        f"\n当前长度：{len(series)}"
        f"\n正确长度：{EXPECTED_N_POINTS}"
    )

    assert fp == EXPECTED_FINGERPRINT, (
        f"\n❌ 数据指纹错误！"
        f"\n当前指纹：{fp}"
        f"\n正确指纹：{EXPECTED_FINGERPRINT}"
    )


    print()
    print("✅ 正式检查点 1 通过！")
    print()
    print(
        f"序列长度 = {EXPECTED_N_POINTS}"
    )
    print(
        f"数据指纹 = {EXPECTED_FINGERPRINT}"
    )


    # --------------------------------------------------------
    # 查看前 10 个聚合值
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("⑩ 聚合序列前 10 个值")
    print("=" * 70)

    print(
        series.head(10)
    )


    # --------------------------------------------------------
    # 查看最后 10 个聚合值
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("⑪ 聚合序列最后 10 个值")
    print("=" * 70)

    print(
        series.tail(10)
    )


    # --------------------------------------------------------
    # 保存序列
    #
    # 后续 EDA / ACF / CV / 模型都直接使用它
    # --------------------------------------------------------

    output_dir = (
        HERE / "student_output"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        output_dir /
        f"series_{SID}.csv"
    )

    series.rename(
        TARGET_COL
    ).to_csv(
        output_path,
        header=True
    )


    print()
    print("=" * 70)
    print("阶段 1 完成")
    print("=" * 70)

    print(
        "聚合后的时间序列已保存到："
    )

    print(
        output_path
    )


# ============================================================
# 6. 程序入口
# ============================================================

if __name__ == "__main__":
    main()