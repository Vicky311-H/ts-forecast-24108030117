# -*- coding: utf-8 -*-

from pathlib import Path
import zipfile


SID = "24108030117"

ROOT = Path(__file__).resolve().parent

ZIP_PATH = ROOT / f"{SID}_results.zip"

TOP_DIR = f"{SID}_results"


# ============================================================
# 必须包含的文件
# ============================================================

REQUIRED_FILES = [

    ROOT / f"hw_{SID}.py",

    ROOT / f"hw_{SID}_stage2.py",

    ROOT / f"hw_{SID}_stage3.py",

    ROOT / f"hw_{SID}_stage4.py",

    ROOT / f"hw_{SID}_stage5.py",

    ROOT / f"hw_{SID}_stage6.py",

    ROOT / f"hw_{SID}_stage7.py",

    ROOT
    / "homework"
    / f"hw_{SID}.md",

    ROOT
    / "student_output"
    / f"series_{SID}.csv",
]


# ============================================================
# 必须包含的结果目录
# ============================================================

REQUIRED_DIRS = [

    ROOT
    / "student_output"
    / f"stage{i}"

    for i in range(
        2,
        8
    )
]


# ============================================================
# 生成 ZIP 内部路径
# ============================================================

def arcname_for(
    path: Path
) -> str:

    relative = (
        path.relative_to(
            ROOT
        )
    )

    return str(
        Path(
            TOP_DIR
        )
        / relative
    ).replace(
        "\\",
        "/"
    )


# ============================================================
# 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 如果之前已经存在旧 ZIP，先删除
    # 避免把旧结果误认为本次新结果
    # --------------------------------------------------------

    if ZIP_PATH.exists():

        ZIP_PATH.unlink()


    # --------------------------------------------------------
    # 检查关键文件是否齐全
    # --------------------------------------------------------

    missing = [

        str(
            p.relative_to(
                ROOT
            )
        )

        for p in REQUIRED_FILES

        if not p.is_file()
    ]


    missing += [

        str(
            p.relative_to(
                ROOT
            )
        )

        for p in REQUIRED_DIRS

        if not p.is_dir()
    ]


    # --------------------------------------------------------
    # 有缺失就停止
    # --------------------------------------------------------

    if missing:

        print(
            "❌ 未压缩：发现缺失项："
        )

        for item in missing:

            print(
                "  -",
                item
            )

        raise SystemExit(
            1
        )


    # --------------------------------------------------------
    # 防止同一个文件重复添加
    # --------------------------------------------------------

    added = set()


    # --------------------------------------------------------
    # 创建 ZIP
    # --------------------------------------------------------

    with zipfile.ZipFile(

        ZIP_PATH,

        mode="w",

        compression=zipfile.ZIP_DEFLATED,

        compresslevel=9,

    ) as zf:


        # ====================================================
        # 添加主程序、任务书、聚合序列
        # ====================================================

        for file_path in REQUIRED_FILES:

            arcname = arcname_for(
                file_path
            )


            zf.write(
                file_path,
                arcname
            )


            added.add(
                arcname
            )


        # ====================================================
        # 添加 stage2 ~ stage7 全部结果
        # ====================================================

        for folder in REQUIRED_DIRS:


            for file_path in sorted(
                folder.rglob(
                    "*"
                )
            ):


                # 只压缩文件
                if not file_path.is_file():

                    continue


                # 跳过 Python 缓存
                if "__pycache__" in file_path.parts:

                    continue


                if file_path.suffix.lower() in {
                    ".pyc",
                    ".pyo",
                }:

                    continue


                arcname = arcname_for(
                    file_path
                )


                if arcname not in added:

                    zf.write(
                        file_path,
                        arcname
                    )

                    added.add(
                        arcname
                    )


    # --------------------------------------------------------
    # ZIP 完整性检查
    # --------------------------------------------------------

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zf:

        bad_file = (
            zf.testzip()
        )

        file_count = len(
            zf.namelist()
        )


    # --------------------------------------------------------
    # 如果 ZIP 损坏，删除
    # --------------------------------------------------------

    if bad_file is not None:

        ZIP_PATH.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"ZIP 校验失败，"
            f"损坏文件：{bad_file}"
        )


    # --------------------------------------------------------
    # 最终结果
    # --------------------------------------------------------

    print()

    print(
        "✅ 压缩完成"
    )

    print(
        "压缩包：",
        ZIP_PATH
    )

    print(
        "文件数：",
        file_count
    )

    print(
        "大小：",
        f"{ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB"
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()