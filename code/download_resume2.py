# -*- coding: utf-8 -*-
"""稳定下载器 v2：基于 Content-Length 大小校验 + 断点续传"""
import urllib.request, os, sys, time

def download_with_resume(url, outpath, timeout=180):
    headers = {'User-Agent': 'Mozilla/5.0'}
    existing = os.path.getsize(outpath) if os.path.exists(outpath) else 0
    if existing > 0:
        headers['Range'] = f'bytes={existing}-'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total = int(r.headers.get('Content-Length', 0)) + existing
            mode = 'ab' if existing > 0 else 'wb'
            with open(outpath, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            now_size = os.path.getsize(outpath)
            print(f'  > 当前 {now_size/1024/1024:.1f}MB / 预计 {total/1024/1024:.1f}MB', flush=True)
            return now_size, total
    except Exception as e:
        now_size = os.path.getsize(outpath) if os.path.exists(outpath) else 0
        print(f'  中断: {e} (已 {now_size/1024/1024:.1f}MB)', flush=True)
        return now_size, None

def run(url, outpath, max_attempts=30):
    for i in range(max_attempts):
        print(f'尝试 {i+1}/{max_attempts}...', flush=True)
        size, total = download_with_resume(url, outpath)
        if total is not None and size >= total:
            print(f'✓ 完成: {size/1024/1024:.1f}MB')
            return True
        time.sleep(2)
    print('已达最大尝试次数')
    return False

if __name__ == '__main__':
    ok = run(sys.argv[1], sys.argv[2])
    print('FINAL:', 'OK' if ok else 'FAIL')
