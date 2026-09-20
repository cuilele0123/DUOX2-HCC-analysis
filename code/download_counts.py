# -*- coding: utf-8 -*-
"""批量下载 TCGA-LIHC STAR-Counts 表达文件 + 临床数据"""
import json, os, time, urllib.request, concurrent.futures

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(_ROOT, 'data')
CNT_DIR = os.path.join(BASE, 'counts')
os.makedirs(CNT_DIR, exist_ok=True)

# 读取清单
rows = []
with open(os.path.join(BASE, 'lihc_files_list.tsv')) as f:
    header = f.readline()
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4:
            rows.append({'file_id': parts[0], 'file_name': parts[1],
                         'submitter_id': parts[2], 'sample_type': parts[3]})

print(f'待下载: {len(rows)} 个文件')

def download_one(row):
    out = os.path.join(CNT_DIR, row['file_id'] + '.tsv')
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return (row['submitter_id'], 'skip')
    url = f"https://api.gdc.cancer.gov/data/{row['file_id']}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with open(out, 'wb') as f:
                f.write(data)
            return (row['submitter_id'], f"{len(data)//1024}KB")
        except Exception as e:
            if attempt == 2:
                return (row['submitter_id'], f'ERR:{e}')
            time.sleep(2)

done = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    for res in ex.map(download_one, rows):
        done += 1
        if done % 50 == 0:
            print(f'  {done}/{len(rows)} 完成', flush=True)

# 统计
files = os.listdir(CNT_DIR)
sizes = [os.path.getsize(os.path.join(CNT_DIR, f)) for f in files]
print(f'下载完成: {len(files)} 个文件, 总大小 {sum(sizes)/1024/1024:.0f} MB')
