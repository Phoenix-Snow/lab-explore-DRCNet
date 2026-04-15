import os
import platform
import time
import signal
import multiprocessing as mp
import psutil


class ProcessGuardian:
    def __init__(self, target_pid, heartbeat_threshold=10):
        self.stop_event = mp.Event()
        self.guardian_process = None
        self.target_pid = target_pid
        self.guardian_pid = os.getpid()
        self.last_beat = mp.Value('d', 0.0)  # 共享：记录主进程最后一次心跳
        self.heartbeat_threshold = heartbeat_threshold
        manager = mp.Manager()
        self.tid_list = manager.list()
        print(f"🛡️ [Guardian] monitor PID: {self.guardian_pid}, guard PID: {self.target_pid}, "
              f"heartbeat threshold: {self.heartbeat_threshold} seconds")

    # ==========================================
    # 主进程调用，🔄 初始化 🛡️ [监护者] 的监护进程
    # ==========================================
    def init(self):
        """🔄 初始化 🛡️ [监护者] 的监护进程"""
        self.guardian_process = mp.Process(target=self._monitor_loop)
        self.guardian_process.daemon = True  # 设为守护进程
        self.guardian_process.start()
        print(f"🛡️ [Guardian] monitor process is started (PID: {self.guardian_process.pid})")

    # ==========================================
    # 主进程调用，⏰ 向 🛡️ [监护者] 发生心跳信号
    # ==========================================
    def record_heartbeat(self):
        """⏰ 向 🛡️ [监护者] 发生心跳信号"""
        # 直接修改共享内存的值
        with self.last_beat.get_lock():
            self.last_beat.value = time.time()

    # ==========================================
    # 主进程调用， 🔄 向 🛡️ [监护者] 汇报线程ID
    # ==========================================
    def record_TID(self, tid, name):
        """🔄 向 🛡️ [监护者] 汇报线程ID"""
        self.tid_list.append({'tid': tid, 'name': name})
        print(f"🛡️ [Guardian] receive TID registration: {name} (TID: {tid})")

    # ==========================================
    # 定时任务，⏰ 监视心跳进程逻辑
    # ==========================================
    def _monitor_loop(self):
        # try:
        #     # 获取被监控的主进程对象
        #     proc = psutil.Process(self.target_pid)
        #     print(f"🛡️ [Guardian] 开始监控 PID: {self.target_pid}")
        # except psutil.NoSuchProcess:
        #     print("🛡️ [Guardian] 目标进程不存在！")
        #     return
        self.record_heartbeat()  # 启动时间
        while not self.stop_event.is_set():
            """⏰ 心跳检测进程逻辑"""
            time.sleep(2)

            if not self._heartbeat_check():
                print(f"🛡️ [监护者] ⚠️ 心跳丢失超过 {self.heartbeat_threshold}秒，执行清理！")
                try:
                    self._kill_process_tree()  # 杀其他进程，一般主进程杀了，这个进程也会被杀
                    self.stop()  # 如果没有，杀自己
                    break
                except Exception as e:
                    raise Exception(f"🛡️ [监护者] ⚡ 进程清理失败: {e}, 尝试使用 Exception 终止")



    # ==========================================
    # BOOL判断，💓 心跳检测
    # ==========================================
    def _heartbeat_check(self):
        if time.time() - self.last_beat.value > self.heartbeat_threshold:
            return False
        else:
            return True

    # ==========================================
    # 进程清理方法，✂️ 监控对象进程树从分支到主干清理
    # ==========================================
    def _kill_process_tree(self):
        print(f"🛡️ [Guardian] 💀 Killing: PID {self.target_pid}")

        # 1. 先杀子进程（防止孤儿）
        for item in self.tid_list:
            tid = item['tid']
            name = item['name']
            try:
                if platform.system() == "Windows":
                    # Windows 下使用 SIGTERM 或者 CTRL_BREAK_EVENT
                    # 注意：os.kill 在 Windows 下只能对进程 ID (PID) 使用，不能对线程 ID (TID) 使用
                    # os.kill(pid, signal.SIGTERM)
                    pass
                else:
                    # Linux/Unix 下使用 SIGKILL
                    os.kill(tid, signal.SIGKILL)
                print(f"🛡️ [Guardian] ✅ thread {name} (TID: {tid}) closed")
            except ProcessLookupError:
                print(f"🛡️ [Guardian] ℹ️ thread {name} (TID: {tid}) already disappeared")
            except PermissionError:
                print(f"🛡️ [Guardian] ❌ thread {name} (TID: {tid}) : no permission")
            except Exception as e:
                print(f"🛡️ [Guardian] ❌ thread {name} closing task error: {e}")

        # 2. 再杀本体（防止僵尸）
        try:
            proc = psutil.Process(self.target_pid)
            proc.kill()
            print(f"🛡️ [Guardian] 💀 Main process (PID {self.target_pid}) closed")
        except psutil.NoSuchProcess:
            print(f"🛡️ [Guardian] ℹ️ Main process (PID {self.target_pid}) already disappeared")
        except Exception as e:
            print(f"🛡️ [Guardian] ❌ Main process (PID {self.target_pid}) closing task error: {e}")

    # ==========================================
    # 自杀程序，🔪 清理掉自身进程
    # ==========================================
    def stop(self):
        self.stop_event.set()
        if self.guardian_process:
            self.guardian_process.join(timeout=2)
            if self.guardian_process.is_alive():
                try:
                    self.guardian_process.kill()
                    print(f"🛡️ [Guardian] 💀 Guardian (PID {self.guardian_pid}) closed")
                except Exception as e:
                    print(f"🛡️ [Guardian] ❌ Guardian (PID {self.guardian_pid}) closing task error: {e}")
            else:
                print(f'️🛡️ [Guardian] ℹ️ Guardian (PID {self.guardian_pid}) is already not working')
                del self.guardian_process
        else:
            print(f'️🛡️ [Guardian] ℹ️ Guardian (PID {self.guardian_pid}) already disappeared')