# MiniCPM-o 2.6：原版 Testbed RTC 无扰动基线

日期：2026-10-10。固定视频 1–25，每视频 5 问；两种输入各 125 题，共 250 次真实 RTC 回答。

| 输入 | 答对/125 | 准确率 | 无效/缺失回答 | 缺失特征包 |
|---|---:|---:|---:|---:|
| 提问前最近原生单帧 | 98/125 | 78.4% | 0/0 | 0/318625 |
| 该帧及紧邻前三帧 | 101/125 | 80.8% | 0/0 | 0/1274375 |

四帧使 4 题由错变对、1 题由对变错；固定子集结果不代表统计显著性。

## 拆分与传输一致性

- 全部 250 题：实际服务器特征 SHA-256 等于原始特征，缺失分片为零。
- 全部 250 题重新运行视觉端，原始帧哈希与连续索引核验通过，重编码特征哈希逐字节复现。
- 独立未拆分 reference 调用上游 `mtmd_helper_eval_chunks`，不用手动特征注入或 RTC。全部 250 题的答案、生成 token IDs 和首步完整 logits 的字节 SHA-256 与拆分 RTC 完全一致。
- 每输入模式抓到 125 个 RTP 时间戳消息组；最大媒体 IP 包 1167 字节，未超过 1500。
- 干净原版 checkout 可直接应用源码补丁，30 项相关检查通过。
- 干净副本额外完成 no_emulation / lossy 各 5 题真实 RTC smoke；10 条原始/置零特征重编码哈希通过。这验证了公开补丁能从原版 orchestrator 启动，仍不是完整受扰动数据集结果。

[逐题答案对照](paired_predictions.csv) · [独立 helper 原始结果](linux_unsplit_reference_full.json) · [250 题逐项 parity](parity.csv) · [重新编码审计](audit/audit.json) · [汇总 CSV](audit/condition_summary.csv)

## 输入、运行时与计时边界

同机 WSL Linux 两个进程共享 RTX 3070；端侧为视觉编码器及 resampler，服务器为语言模型。中间传 F32 embedding，视觉数据走原版 UDP/RTP/SRTP 与 lo 网卡；TCP 仅信令，问题/布局/答案经可靠 DataChannel。使用原版 orchestrator、netemu、抓包和 SQLite 流程。

Q4_K_M 语言模型 / F16 视觉，llama.cpp b11516 CUDA 12.8，贪心解码，标准答案不进入模型提示词。与此前 Windows Vulkan 数值不同，后续网络条件须对照这里的 Linux 基线。

与此前 Windows Vulkan 逐题比较：单帧答案全部相同；四帧仅 `sample_24_3` 由 B 变为 D（标准 D），因此 100/125 变为 101/125。同一 Linux 运行时拆分/未拆分完全一致，不能把跨后端差异归因于 RTC。[后端答案差异记录](backend_answer_differences.json)。

完整基线执行 checkout 同时含此前其他任务的 HTTP 计时、TCP-window 扩展与纳秒抓包设置；HTTP/TCP-window 路径未启用，抓包精度不改变模型或网络 profile。公开补丁只含 MiniCPM 所需修改；另用干净上游副本完成上述真实 RTC smoke，避免依赖未发布的其他任务修改。

| 输入 | 平均端侧编码 | 平均服务器推理（含队列） | 平均元数据登记至完整答案 |
|---|---:|---:|---:|
| single | 0.451s | 0.436s | 1.854s |
| native4 | 1.448s | 1.135s | 3.563s |

上述计时不是端到端延迟；元数据计时不含端侧编码和答案回传。答案一次性回传，未测逐 token TTFT。原版网络报告中的 TTFT/TTLT/stall 字段保留原名，但这里是五道独立问题的整条回答事件，不能当作逐 token 指标。

**本轮 track 只按问题顺序加速传输指定原生单帧/连续四帧；不是持续原生 FPS 流量。**

保留原版每视频之间 20 秒间隔。原时间轴/持续特征流与按提问点加速的正式受扰动口径仍待确定。本报告只证明拆分 RTC 基线与完整 125 题结果；不能声称完整多 profile 连续视频测试已完成。

本次服务器和客户端已停止，18124 监听已关闭，lo 恢复 noqueue；[清理核验](cleanup.json) 已通过。

[源码应用及复现说明](../../docs/testbed_rtc.md) · [四种网络配置预检（每条件仅 5 题）](../testbed_rtc_readiness_20261010/report_zh.md) · [抓包与队列摘要](../testbed_rtc_network_20261010.json)
