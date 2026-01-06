import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor

# 配置参数
metadata_path = "./geoLocalization_schema.csv"
download_dir = "./audios"
num_ids = 100000000000
max_workers = 8

# 创建目标目录
os.makedirs(download_dir, exist_ok=True)

# 读取 metadata.csv 的前 num_ids 个条目
entries = []
with open(metadata_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= num_ids:
            break
        entries.append((row["key"], row["mp3name"]))

# 下载函数
def download_mp3(entry):
    key, mp3name = entry
    url = f"https://archive.org/download/{key}/{mp3name}"
    save_path = os.path.join(download_dir, mp3name)

    # ✅ 如果文件已经存在，跳过下载
    if os.path.exists(save_path):
        print(f"⏩ 已存在，跳过: {mp3name}")
        return

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"✅ 下载成功: {mp3name}")
        else:
            print(f"❌ 下载失败: {mp3name} - 状态码: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 错误下载 {mp3name}: {e}")

# 并发下载
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(download_mp3, entries)
