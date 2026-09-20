# -*- coding: utf-8 -*-
"""多线程分段并发下载（Range 并行拉取，断点续传，最后拼接）
解决 NCBI FTP 单连接速度慢/易断的问题
"""
import urllib.request, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK = 4 * 1024 * 1024  # 4MB/段

def get_size(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get('Content-Length', 0))

def download_range(url, start, end, outpath, retries=6):
    headers = {'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'}
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with open(outpath, 'r+b') as f:
                f.seek(start)
                f.write(data)
            return True
        except Exception as e:
            if i == retries - 1:
                return False
            time.sleep(3)
    return False

def run(url, outpath, n_threads=8, max_retries=3):
    total = get_size(url)
    print(f'文件总大小: {total/1024/1024:.1f}MB')
    if total <= 0:
        print('无法获取大小'); return False
    # 初始化空文件
    if not os.path.exists(outpath) or os.path.getsize(outpath) < total:
        with open(outpath, 'wb') as f:
            f.truncate(total)
    # 计算未完成的段
    chunks = [(s, min(s+CHUNK-1, total-1)) for s in range(0, total, CHUNK)]
    for attempt in range(max_retries):
        # 找未完成段（文件里该段全 0 视为未完成，简化判断用已有大小）
        done_size = 0
        with open(outpath, 'rb') as f:
            done_size = sum(1 for _ in f) - 1  # 粗略
        remaining = []
        with open(outpath, 'rb') as f:
            for s, e in chunks:
                f.seek(s)
                first = f.read(1)
                # 简化：检查段首是否有数据（非 \x00）
                if first == b'\x00':
                    remaining.append((s, e))
        if not remaining:
            print(f'✓ 全部完成: {total/1024/1024:.1f}MB')
            return True
        print(f'第{attempt+1}轮: 待下载 {len(remaining)} 段')
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = {ex.submit(download_range, url, s, e, outpath): (s, e) for s, e in remaining}
            ok_cnt = 0
            for fut in as_completed(futures):
                if fut.result():
                    ok_cnt += 1
        print(f'  本轮完成 {ok_cnt}/{len(remaining)} 段')
        if ok_cnt == len(remaining):
            print(f'✓ 下载完成: {os.path.getsize(outpath)/1024/1024:.1f}MB')
            return True
        time.sleep(2)
    return False

if __name__ == '__main__':
    ok = run(sys.argv[1], sys.argv[2])
    print('FINAL:', 'OK' if ok else 'FAIL')
