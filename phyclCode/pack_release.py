"""
打包 PhyCL-FallDetect 跌倒检测系统及相关代码、数据集为 zip 文件
"""
import os
import shutil
import zipfile
from pathlib import Path

def pack_release():
    # 输出zip文件名
    output_zip = "PhyCL-FallDetect_V1.0.zip"
    
    # 临时打包目录
    pack_dir = Path("_pack_temp")
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir()
    
    # 目标结构
    # PhyCL-FallDetect/
    #   ├── PhyCL-FallDetect/      <- exe及依赖
    #   ├── SisFall/               <- 数据集
    #   │   ├── ADL/
    #   │   └── FALL/
    #   ├── source_code/           <- 源代码
    #   │   ├── app/
    #   │   ├── models/
    #   │   ├── losses/
    #   │   └── ...
    #   ├── sample_input.csv
    #   └── 用户手册.md
    
    root = pack_dir / "PhyCL-FallDetect"
    root.mkdir()
    
    print("1. 复制 PhyCL-FallDetect.exe 及依赖...")
    # 从 dist/PhyCLNetDemo 复制并重命名
    src_dist = Path("dist/PhyCLNetDemo")
    if src_dist.exists():
        dst_dist = root / "PhyCL-FallDetect"
        shutil.copytree(src_dist, dst_dist)
        # 重命名exe
        old_exe = dst_dist / "PhyCLNetDemo.exe"
        new_exe = dst_dist / "PhyCL-FallDetect.exe"
        if old_exe.exists():
            old_exe.rename(new_exe)
    else:
        print(f"  警告: {src_dist} 不存在，跳过")
    
    print("2. 复制 SisFall 数据集...")
    src_sisfall = Path("../data/SisFall")
    if src_sisfall.exists():
        dst_sisfall = root / "SisFall"
        dst_sisfall.mkdir()
        # 只复制 ADL 和 FALL 目录
        for subdir in ["ADL", "FALL"]:
            src_sub = src_sisfall / subdir
            if src_sub.exists():
                shutil.copytree(src_sub, dst_sisfall / subdir)
                print(f"  复制 {subdir}/")
    else:
        print(f"  警告: {src_sisfall} 不存在，跳过")
    
    print("3. 复制源代码...")
    src_code = root / "source_code"
    src_code.mkdir()
    
    # 复制代码目录
    code_dirs = ["app", "models", "losses"]
    for d in code_dirs:
        src = Path(d)
        if src.exists():
            # 排除 __pycache__
            shutil.copytree(src, src_code / d, 
                          ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"  复制 {d}/")
    
    # 复制主要Python文件
    py_files = [
        "PhyCL_Net_experiments.py",
    ]
    for f in py_files:
        src = Path(f)
        if src.exists():
            shutil.copy2(src, src_code / f)
            print(f"  复制 {f}")
    
    # 复制requirements
    if Path("requirements-demo.txt").exists():
        shutil.copy2("requirements-demo.txt", src_code / "requirements.txt")
        print("  复制 requirements.txt")
    
    print("4. 复制样例文件...")
    if Path("sample_input.csv").exists():
        shutil.copy2("sample_input.csv", root / "sample_input.csv")
    
    print("5. 复制用户手册...")
    if Path("PhyCLNetDemo_说明书.md").exists():
        shutil.copy2("PhyCLNetDemo_说明书.md", root / "用户手册.md")
    
    print("6. 创建 zip 文件...")
    # 删除旧的zip
    if os.path.exists(output_zip):
        os.remove(output_zip)
    
    # 创建zip
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in root.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(pack_dir)
                zf.write(file_path, arcname)
                
    print("7. 清理临时目录...")
    shutil.rmtree(pack_dir)
    
    # 获取文件大小
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n完成! 输出文件: {output_zip} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    pack_release()
