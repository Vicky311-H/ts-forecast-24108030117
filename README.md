# 时间序列预测课程作业 — 学生材料包

本包包含完成《时间序列预测与可视化》课程作业所需的**全部材料**，无需联网、无需从别处索取文件。
请先读完本 README，再按「使用流程」逐步操作。

---

## 一、文件清单与用途

| 文件 | 用途 | 何时用 |
|---|---|---|
| `flow.csv` | 原始交通流数据（约 50 万条过车事件，2018-02） | 全程，代码读它 |
| `Day5_TimeSeries_Forecast_DeepTheory.html` | **教学课件**（三代范式、平稳性、ACF/PACF、现代线性模型、TSFM、评估指标） | 第 0 步起，卡住时回查 |
| `timeseries_flow_code.py` | **示例代码**：教学示例的完整预测流水线（聚合→EDA→平稳性→ACF/PACF→分解→基线→现代线性模型→滚动起源 CV→残差诊断→权重图）。**这是你写作业的参考范本**，不是你的任务书 | 写代码时对照抄改 |
| `hw_task_selfservice.py` | **任务书生成器（学生版）**：输入你的学号，生成专属于你的任务书（目标变量/方向/车道/粒度/天数/窗口/步长/CV 折数/种子/指纹/锚点值） | 第 1 步 |
| `homework/学生作业指导书.docx` | **操作指南**：第 0–10 步怎么做、课件章节速查、示例代码函数地图、基准数字、排错表、AI 使用边界、评分标准 | 不会做时逐条查 |
| `homework/学生作业报告模板.docx` | **提交模板**：照着填即可，含 50 个待填书签（Tab 跳转），填完约 2 页 | 最后填 |

> ⚠️ **不要**找老师要 `hw_spec_generator.py`（教师审计版，含全班批量与核验功能，学生不需要也拿不到），也不要把本包里的文件随意改名——任务书生成器是按文件名读 `flow.csv` 的。

---

## 二、运行环境准备

需要 **Python 3.10+**。在命令行（Windows 的 PowerShell / macOS 的终端）里执行：

```bash
# 1) 安装依赖（只需一次）
pip install pandas numpy matplotlib scikit-learn statsmodels

# 2) 验证能跑通示例代码（约 1–2 分钟，CPU 即可）
python timeseries_flow_code.py
```

跑通后会在 `output/` 下生成 9 张图与 `results.json`——这证明你的环境没问题，也帮你理解示例结果。

> 包内未附带 `output/`（那是教师的运行结果）。你自己跑一遍 `timeseries_flow_code.py` 即可得到。

---

## 三、使用流程（五步）

1. **读课件**：打开 `Day5_TimeSeries_Forecast_DeepTheory.html`，重点看第 5 章（平稳性与 ACF/PACF）、第 6 章（三代范式与前沿模型）、第 7–8 章（评估指标与滚动起源 CV）。
2. **领任务书**：运行
   ```bash
   python hw_task_selfservice.py --sid <你的学号>
   ```
   例如 `python hw_task_selfservice.py --sid 202504003`。屏幕上会打印你的专属参数，并自动生成 `homework/hw_<你的学号>.md` 任务书。**务必保管好这份任务书里的「数据指纹」和「锚点值」**——它们会出现在你的报告中，且老师能复算校验。
3. **读指导书**：打开 `homework/学生作业指导书.docx`，按第 0–10 步操作。把 `timeseries_flow_code.py` 当作范本，按你任务书的参数改写。
4. **做实验、留痕迹**：用 git 记录过程（要求 ≥8 次提交、跨 ≥3 天）。保存你的预测图、逐折指标表、误差时段分析图。
5. **填报告**：打开 `homework/学生作业报告模板.docx`，在书签处逐格填写，导出 PDF 提交。

---

## 四、必须知道的「红线」（与成绩直接相关）

- **数据指纹**（`hw_xxxxxx` 12 位十六进制）和**锚点值**（你口径下折 1 的 Seasonal Naive MASE）必须写进报告，且**对得上**老师用同一学号复算的结果。对不上 = 报告无效。
- **禁止特征泄露**：不能用未来信息预测过去；测试折不能参与归一化/标准化/模型训练。
- **禁止在测试集上调参**：所有超参数只能在训练折内确定。
- **AI 可用，但必须标注且必须自己跑通**：你可以问 AI 概念、要代码片段，但最终代码必须能复现你的数字；报告中「三个关键决定」要写清**放弃的理由**，这是 AI 最容易被识破的环节。
- **反事实实验**：指导书要求你人为制造一次泄露并报告成对数字，用来证明你真的理解了泄露的危害——别漏做。

---

## 五、提交清单

1. `学生作业报告模板.docx`（或导出 PDF），填完整，含指纹与锚点值。
2. 你的作业代码（`.py`），**可复现**（固定随机种子，换种子/换指定折重跑数字稳定）。
3. `homework/hw_<你的学号>.md` 任务书。
4. 过程留痕（`git log` 截图或导出，≥8 次提交、跨 ≥3 天）。

如有疑问，先查 `学生作业指导书.docx` 的排错表；仍未解决再问老师。祝你顺利。

## How to Run

Run the experiment stages in order:

1. Stage 1: data preprocessing and fingerprint verification
2. Stage 2: Seasonal Naive baseline
3. Stage 3: Ridge-AR
4. Stage 4: NLinear
5. Stage 5: residual diagnostics and leakage counterfactual
6. Stage 6: time-block error analysis and Direct vs Recursive comparison
7. Stage 7: final summary

Key configuration:

- Student ID: 24108030117
- Target: Gap (s)
- Direction: South
- Lane: SB_NS
- Aggregation: 1 hour
- Sequence length: 168
- Look-back: 60
- Seasonal period: 24
- Horizon: 4
- Fingerprint: e461b6a28e6d

## Repository Structure

- `hw_24108030117.py`: main preprocessing and experiment entry
- `hw_24108030117_stage2.py`: Seasonal Naive baseline and rolling-origin evaluation
- `hw_24108030117_stage3.py`: Ridge-AR model
- `hw_24108030117_stage4.py`: NLinear model
- `hw_24108030117_stage5.py`: residual diagnostics and leakage counterfactual
- `hw_24108030117_stage6.py`: time-block analysis and Direct vs Recursive comparison
- `hw_24108030117_stage7.py`: final experiment summary
- `student_output/`: generated experiment outputs
- `RESULTS.md`: summary of the main experimental results

## Key Configuration

- Look-back: 60
- Seasonal period: 24
- Forecast horizon: 4
- Rolling-origin CV folds: 5
- Fingerprint: `e461b6a28e6d`