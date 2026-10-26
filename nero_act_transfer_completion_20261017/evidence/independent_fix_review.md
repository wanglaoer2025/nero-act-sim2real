# 独立复核与一次修正

独立 reviewer `/root/completion_stop_review` 只读检查了本次门控、运行时取消、信号和监督器修复，未连接硬件。提出两项 Important：运行时继承的 rclpy 默认 SIGINT 会先关闭 ROS；监督器 SIGHUP/SIGTERM 默认退出会跳过 finally。另要求加强接收进程记录后的存活、独立 session 及父进程退出后的核对。

一次修正：ActSession 构造完成后、任何目标之前卸载 rclpy 信号处理，保留 Python KeyboardInterrupt 到 finally；监督器将 HUP/TERM 转入 cleanup，cleanup 期间忽略重复终止信号；接收进程使用独立 session、禁用终端输入，启动记录后校验存活、session及禁止发送配置，并标记仍需会话结束后的真实反馈复核。

验证：实际 ActSession 的隔离域43 SIGINT 测试在修复前复现 ROS 失效；修复后 ROS 存活，先永久锁定门控，再取得假驱动关门应答，CAN不变。见 runtime_cleanup_red/report.json、runtime_cleanup_green/report.json。HUP/TERM 假监督器测试在修复前失败，修复后都执行 finally 并保留父进程退出后存活的独立接收子进程。完整38项测试通过。修正由作者验证，未声称第二轮独立复核。
