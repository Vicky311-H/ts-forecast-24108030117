# -*- coding: utf-8 -*-
"""
学号任务书自助生成器（学生版）—— 你的专属作业任务书
============================================================
配套教学文件: Day5_TimeSeries_Forecast_DeepTheory.html（第 10 章 课后作业）

【怎么用】
    python hw_task_selfservice.py --sid 202504003
    python hw_task_selfservice.py --sid 202504003 --csv flow.csv

会在当前目录生成 homework/hw_<你的学号>.md，那就是你的专属任务书。

【这个脚本在做什么】
把"学号"确定性地映射成一份**个性化任务书**：数据口径、预测步长、交叉验证折数、
随机种子、以及一个只属于你的**数据指纹**和**锚点值**。
每个同学的口径组合都不一样，所以**不存在通用答案**。

【三条重要规则，请务必遵守】
 1. **不要修改本脚本的任何一行。** 教师手里有原始副本，会用你的学号复算
    你的数据指纹与锚点值；对不上，说明你的报告里的数字不是你自己跑出来的。
 2. **不要查阅同学的学号去跑他们的任务书。** 即使你算出来了，那个数字也填不进
    你的报告——教师按<b>你的</b>学号复算，对不上只会暴露你自己。
 3. 你的任务书里给出的<b>锚点值</b>，就是"你自己必做的那个朴素基线"在第 1 折上的 MASE。
    它不是机密，而是用来证明"报告里的数字确实出自你自己的代码"的校验点。

【为什么发放到个人手里】
一是不给任何"统一题目"的口子，AI 无法预判你的参数组合；
二是口径与你的报告严格绑定，抄袭者的报告必然对不上你的指纹。
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 口径候选池
# 目标变量：故意混入比 speed 更"难"的两个
TARGETS = {
    "speed":   ("Speed (mph)", 10.0, 80.0, "平均车速 (mph)"),
    "headway": ("Headway (s)", 0.3, 60.0, "车头时距 (s)"),
    "gap":     ("Gap (s)", 0.3, 60.0, "车头空距 (s)"),
}
DIRECTIONS = {1: "North(北行)", 2: "South(南行)"}
LANES = {1: ["NB_NS", "NB_MID", "NB_OS"], 2: ["SB_NS", "SB_MID", "SB_OS"]}
GRANS = {"10min": ("10min", 1 / 6, 144), "15min": ("15min", 0.25, 96),
         "30min": ("30min", 0.5, 48), "1h": ("1h", 1.0, 24)}
N_DAYS_OPTS = [8, 7, 6, 5]                          # 用多少天（从末端往回取）
LOOK_BACK_MULTS = [1.0, 1.5, 2.0, 2.5]                # 窗口 = 倍数 x 一个日周期
HORIZON_CHOICES = [round(0.5 * i, 1) for i in range(1, 17)]   # 0.5 ~ 8.0 小时
CV_FOLDS = [3, 4, 5]
CLIP_HI_MULTS = [0.90, 0.95, 1.00, 1.10]             # 异常上界档位
CLIP_LO_MULTS = [1.00, 1.25, 1.50]                   # 异常下界档位（更严格）
EXTRA_TASKS = [
    "必须做一次特征消融：去掉滚动标准差特征，观察 MASE 变化并解释原因",
    "必须画出一张『误差随预测步长(1~h)的衰减曲线』，比较递归式与直接式的差别",
    "必须给出 Seasonal Naive 与最优模型的 Diebold-Mariano 检验，并解释 p 值含义",
    "必须逐折做残差 Ljung-Box 检验，说明残差是否还有结构、若有则下一步加什么特征",
    "必须画出误差的时段分布（按小时或工作日/周末），指出最差时段并归因",
]


# ============================================================================
# 核心：无状态派生（学生版与教师版共用同一份实现，保证口径绝不漂移）
# ============================================================================
def seed_from_sid(sid: str) -> int:
    return int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16)


def spec_from_sid(sid: str) -> dict:
    """学号 -> 确定性任务书。**无状态**：只依赖学号本身，
    因此教师复算与你自助运行的结果必然完全一致。"""
    rng = np.random.default_rng(seed_from_sid(sid))
    target = list(TARGETS)[int(rng.integers(len(TARGETS)))]
    col, lo0, hi0, unit = TARGETS[target]
    direction = int(rng.choice([1, 2]))
    scope = "ALL" if rng.random() < 0.4 else LANES[direction][int(rng.integers(3))]
    gran = list(GRANS)[int(rng.integers(len(GRANS)))]
    n_days = int(rng.choice(N_DAYS_OPTS))
    lb_mult = float(rng.choice(LOOK_BACK_MULTS))
    horizon_hours = float(rng.choice(HORIZON_CHOICES))
    cv_folds = int(rng.choice(CV_FOLDS))
    extra = EXTRA_TASKS[int(rng.integers(len(EXTRA_TASKS)))]
    clip_lo = round(lo0 * float(rng.choice(CLIP_LO_MULTS)), 4)
    clip_hi = round(hi0 * float(rng.choice(CLIP_HI_MULTS)), 4)
    rule, g_hours, period = GRANS[gran]
    return {
        "sid": str(sid), "target": target, "col": col, "unit": unit,
        "direction": direction, "scope": scope, "gran": gran, "rule": rule,
        "gran_hours": g_hours, "period": period, "n_days": n_days,
        "look_back": int(round(lb_mult * period)), "look_back_mult": lb_mult,
        "horizon": max(1, int(round(horizon_hours / g_hours))),
        "horizon_hours": horizon_hours, "cv_folds": cv_folds,
        "clip_lo": clip_lo, "clip_hi": clip_hi, "extra": extra,
        "seed": seed_from_sid(sid) % 100000,
    }


def build_series(csv_path, sp):
    """按任务书口径聚合序列（与课件第 1 章同一套逻辑）。"""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Direction"] == sp["direction"]]
    if sp["scope"] != "ALL":
        df = df[df["Lane Name"] == sp["scope"]]
    v = df[sp["col"]].astype(float)
    df = df.loc[v.between(sp["clip_lo"], sp["clip_hi"])]
    days = sorted(df["Date"].dt.date.unique())
    if sp["n_days"] < len(days):
        keep = set(days[-sp["n_days"]:])          # 从末端往回取若干天
        df = df[df["Date"].dt.date.isin(keep)]
    return df.set_index("Date")[sp["col"]].resample(sp["rule"]).mean().dropna().values.astype(float)


def fingerprint(series) -> str:
    """数据指纹：序列四舍五入 6 位后做 SHA256 取前 12 位。"""
    return hashlib.sha256(np.round(series, 6).tobytes()).hexdigest()[:12]


def anchor_value(series, sp):
    """锚点值：Seasonal Naive 在第 1 折上的 MASE。
    这是你必做的基线，所以它本身不是机密——它的作用是让教师能校验你报告里的数字。"""
    L, H, P = sp["look_back"], sp["horizon"], sp["period"]
    min_train = L + H + P
    if len(series) < min_train + H + 1:
        return None
    tr = series[:min_train]
    scale = float(np.mean(np.abs(np.diff(tr))))
    pred = np.resize(tr[-P:], H)
    return float(np.mean(np.abs(series[min_train:min_train + H] - pred)) / scale)


# ============================================================================
# 任务书模板
# ============================================================================
def task_sheet(sp, n_points, fp, anc):
    scope_txt = "该方向全部 3 条车道" if sp["scope"] == "ALL" else f"仅车道 {sp['scope']}"
    days_txt = "全部有效天数" if sp["n_days"] == 8 else f"仅最后 {sp['n_days']} 天"
    return f"""# 课后作业任务书 —— 学号 {sp['sid']}

> 由 `hw_task_selfservice.py` 依据你的学号确定性生成。每个人的口径与参数组合不同，
> 因此**不存在通用答案**。请勿与他人交换代码、结果或任务书。

## 一、数据口径（不可更改）
| 项目 | 你的取值 |
|---|---|
| 方向 | {DIRECTIONS[sp['direction']]} |
| 车道范围 | {sp['scope']}（{scope_txt}） |
| 目标变量 | `{sp['col']}`（{sp['unit']}） |
| 异常值区间 | [{sp['clip_lo']}, {sp['clip_hi']}]，区间外视为传感器异常需剔除 |
| 聚合粒度 | {sp['gran']}（每天 {sp['period']} 个时间箱） |
| 使用天数 | {sp['n_days']} 天（{days_txt}） |
| 序列长度 | {n_points} 点 |
| **数据指纹** | `{fp}` |

- 跨天的无数据时间箱直接丢弃；各天内部按时间顺序首尾相接。
- 指纹算法：把上述序列四舍五入到 6 位小数后做 SHA256，取前 12 位。
  **报告与代码注释中必须出现 `{fp}`**，它证明你确实处理了属于自己的数据。

## 二、建模任务（不可更改）
| 项目 | 你的取值 |
|---|---|
| 输入窗口 look_back | {sp['look_back']} 步（= {sp['look_back_mult']} 个日周期） |
| 季节周期 period | {sp['period']} 步 |
| 预测步长 horizon | {sp['horizon']} 步（= 未来 {sp['horizon_hours']} 小时） |
| 滚动起源交叉验证 | {sp['cv_folds']} 折 + 1 个末端保留折 |
| 随机种子 | {sp['seed']} |
| **附加任务** | {sp['extra']} |

### 必做模型
- **模型 A `Ridge-AR`**：窗口 → horizon 的线性回归。必须报告 `alpha` 的选择过程
  （只能用**训练段尾部**做内层验证，不许用测试集调参）。
- **模型 B `NLinear`**：实例归一化（减窗口末值建模，预测后加回）。
  特征至少含：滞后 1/2/3 步、滞后一个完整周期、过去半周期滚动均值与滚动标准差、时段指示变量。
- **必备基线 `Seasonal Naive`**：周期 = {sp['period']}，它是你必须打败的检察官。

### 评估协议
- 指标：**MASE**（主）+ MAE、RMSE（辅）。MASE 分母只能用**训练段**朴素误差。
- 划分：滚动起源交叉验证 {sp['cv_folds']} 折 + 末端保留折；
  报告<b>每折指标与均值 ± 标准差</b>，不要只报均值。
- 建议直接式多步预测；若用递归式，必须绘制误差随步长的衰减曲线。

## 三、必须体现的思考痕迹（三项，缺一扣分）
1. **反事实实验**：人为制造一次数据泄露（例如标准化改用全序列统计量 / 滚动窗口右移 1 格），
   报告「正确 MASE → 泄露后 MASE」这对数字并解释涨幅来源。
2. **三个关键决定 + 放弃理由**：你做了哪 3 个关键决定？每个决定的备选方案是什么、
   你为什么放弃了它？（例如"周期我选了 X 而不是 Y，因为……"）
3. **逐折残差诊断**：做 Ljung-Box 检验（<b>逐折</b>做，不许把各折残差汇集），
   说明残差是否还有结构、若有则下一步该加什么特征。

## 四、提交物
| 提交物 | 硬性要求 |
|---|---|
| git 仓库 | ≥8 次提交，时间跨度 ≥3 天；提交信息含<b>学号 + 关键指标</b>；每阶段分别提交 |
| 报告 ≤2 页 | 必含：数据指纹 `{fp}`、锚点值、三个决定 + 放弃理由、反事实成对数字、逐折残差诊断、误差时段分析（≤800 字） |
| 图表 ≥4 张 | 序列 + ACF、CV 各折预测对比、残差诊断、误差随步长（或消融） |
| 课堂抽查 | 教师随机指定某一折，要求现场运行并回答"你为什么没用 SARIMA / Transformer" |

### 锚点值（校验用）
> **第 1 折 Seasonal Naive 的 MASE = {round(anc, 4) if anc is not None else 'N/A'}**

这不是机密——它就是你必做的那个基线数字。写入报告，用于证明数字出自你自己的代码。

## 五、诚信条款
- 允许用 AI 查文档、解释概念、debug 自己的报错；<b>必须标注</b>协助点并接受追问。
- 不允许让 AI 直接生成核心建模代码或整篇报告。
- <b>判定标准：能被你当场解释的才算你的。</b>
- 报告中的每个数字必须能由你的仓库代码复现；教师会换种子、随机指定折重跑校验。
"""


# ============================================================================
# 学生入口
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="学号任务书自助生成器（学生版）")
    ap.add_argument("--sid", required=True, help="你的学号")
    ap.add_argument("--csv", default=os.path.join(HERE, "flow.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "homework"))
    args = ap.parse_args()

    sp = spec_from_sid(args.sid)
    series = build_series(args.csv, sp)
    fp = fingerprint(series)
    anc = anchor_value(series, sp)

    print("=" * 68)
    print(f" 学号 {sp['sid']} 的专属任务书")
    print("=" * 68)
    print(f"  目标变量   {sp['col']}（{sp['unit']}），异常区间 [{sp['clip_lo']}, {sp['clip_hi']}]")
    print(f"  方向/车道  {DIRECTIONS[sp['direction']]} / {sp['scope']}")
    print(f"  粒度/天数  {sp['gran']} / {sp['n_days']} 天   -> 序列 {len(series)} 点")
    print(f"  窗口/步长  look_back {sp['look_back']} -> horizon {sp['horizon']}"
          f"（未来 {sp['horizon_hours']} 小时）")
    print(f"  CV 折数    {sp['cv_folds']} 折 + 末端保留     随机种子 {sp['seed']}")
    print(f"  数据指纹   {fp}")
    print(f"  锚点值     折1 SeasonalNaive MASE = {round(anc, 4) if anc is not None else 'N/A'}")
    print(f"  附加任务   {sp['extra']}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"hw_{sp['sid']}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(task_sheet(sp, len(series), fp, anc))
    print("\n  任务书已生成：" + path)
    print("  请仔细阅读，并把它作为你的作业依据。\n")


if __name__ == "__main__":
    main()
