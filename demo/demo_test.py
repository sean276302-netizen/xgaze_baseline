import os
import multiprocessing
import subprocess
import time
import webbrowser
import keyboard  # 使用 keyboard 库监听键盘事件
import threading
import sys

def run_script(script_name):
    subprocess.call(["python", script_name])

# 打开 HTML 文件
def open_html():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")  # HTML 文件路径
    webbrowser.open(html_path)  # 使用默认浏览器打开 HTML 文件

def monitor_output(process, queue):
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        print(line)
        if "detect one face." in line:
            queue.put(True)
            break

if __name__ == "__main__":
    # 创建用于进程间通信的队列
    queue = multiprocessing.Queue()
    
    # 启动 process1 并确保无缓冲输出
    process1 = subprocess.Popen(
        ["python", "-u", "./main.py"],  # 使用 -u 参数禁用缓冲
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=0
    )

    # 创建监控线程
    monitor_thread = threading.Thread(target=monitor_output, args=(process1, queue))
    monitor_thread.daemon = True
    monitor_thread.start()

    # 等待检测到人脸
    try:
        queue.get(timeout=30)  # 等待最多30秒
        # 直接在主进程中打开 HTML，避免多进程问题
        open_html()
        print("成功检测到人脸，已启动浏览器")
    except multiprocessing.queues.Empty:
        print("等待人脸检测超时")
        process1.terminate()
        sys.exit(1)

    # 继续运行主程序
    try:
        while True:
            if keyboard.is_pressed("esc"):  # 检测是否按下 Esc 键
                print("检测到 Esc 键，正在退出...")
                break
            time.sleep(0.1)  # 避免 CPU 占用过高
    except KeyboardInterrupt:
        print("程序被中断。")

    # 终止进程 1
    if process1.poll() is None:  # 使用 poll() 检查 Popen 进程是否还在运行
        process1.terminate()
        process1.wait()
        print("进程 1 已终止。")

    print("程序已退出。")