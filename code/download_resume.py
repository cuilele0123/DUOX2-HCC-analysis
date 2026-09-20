# -*- coding: utf-8 -*-
"""用 Python 分块下载 + 断点续传，更稳定地获取 GEO 队列"""
import urllib.request, os, sys, time

def download_with_resume(url, outpath, timeout=120):
    """断点续传下载"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    # 已有部分
    if os.path.exists(outpath):
        headers['Range'] = f'bytes={os.path.getsize(outpath)}-'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total = int(r.headers.get('Content-Length', 0)) + (os.path.getsize(outpath) if os.path.exists(outpath) else 0)
            print(f'  Content-Range/Size: {r.headers.get("Content-Range", "unknown")} 预计总 {total/1024/1024:.0f}MB')
            mode = 'ab' if os.path.exists(outpath) else 'wb'
            with open(outpath, mode) as f:
                while True:
                    chunk = r.read(1 << 20)  # 1MB
                    if not chunk:
                        break
                    f.write(chunk)
            return True
    except Exception as e:
        print(f'  下载中断: {e} (已下载 {os.path.getsize(outpath)/1024/1024:.1f}MB)')
        return False

def download_until_done(url, outpath, max_attempts=10):
    for i in range(max_attempts):
        print(f'尝试 {i+1}/{max_attempts}...')
        if download_with_resume(url, outpath):
            # 验证 gzip
            import gzip
            try:
                with gzip.open(outpath, 'rb') as f:
                    f.read(1)
                print(f'  ✓ 完成: {os.path.getsize(outpath)/1024/1024:.1f}MB')
                return True
            except Exception as e:
                print(f'  gzip 校验失败: {e}，继续重试')
                time.sleep(3)
        time.sleep(3)
    return False

if __name__ == '__main__':
    url = sys.argv[1]
    out = sys.argv[2]
    ok = download_until_done(url, out)
    print('FINAL:', 'OK' if ok else 'FAIL', os.path.getsize(out)/1024/1024 if os.path.exists(out) else 0, 'MB')
