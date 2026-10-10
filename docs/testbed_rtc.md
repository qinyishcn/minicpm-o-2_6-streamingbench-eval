# MiniCPM-o 2.6：原版 Testbed RTC 接入与复现

本轮在 WSL Ubuntu 本机 local server 模式中，将视觉编码器和 resampler 放在
客户端进程，将语言模型放在服务器进程。两个进程共享 RTX 3070，但经过真实
loopback 网卡的原版 UDP/RTP/SRTP 传输 F32 中间视觉特征。
TCP/18124 只做信令；可靠 DataChannel 传输问题、特征布局元数据和答案。

这是 Q4_K_M 语言模型 / F16 视觉编码器，llama.cpp b11516 CUDA 12.8 的视觉问答
实验，不是官方 BF16 omni 模型的完整复现。与此前 Windows Vulkan 的离线擦除
结果分别报告，同一网络实验只能与同一 Linux RTC 基线比较。

## 当前证据与范围

- [完整无扰动基线及逐题核验](../results/testbed_rtc_baseline_20261010/report_zh.md)：
  单帧、连续四帧各 25 视频 / 125 题。
- [四种原版 profile 的小样本诊断](../results/testbed_rtc_readiness_20261010/report_zh.md)：
  四帧、每种仅视频 1 的 5 道题。不能作为完整数据集准确率。
- [抓包、tc 与 UDP 计数摘要](../results/testbed_rtc_network_20261010.json)：原始抓包和
  SQLite 日志的大小及 SHA-256 保留在此，完整文件留在本机。
- [原版源码补丁](../patches/minicpm_original_testbed.patch)：保留 orchestrator、
  信令、RTC 收发、netemu profile 和捕获/日志，增加模型 codec、特征分片与缺包置零。

当前 track 仅按提问顺序传输指定的单帧 / 四帧特征，源帧来自提问前最近帧及
紧邻的前三帧，无时间抽样；它不是持续原生 FPS 视频流量。
原时间轴/持续模型特征流与按提问点加速的正式受扰动批次，仍需确定回放口径。
不能把这批预检称为完整连续视频网络性能测试。

原版 Liquid 的 RTP 载荷上限为 1100 字节。保留这一预算后，每包有 1080 字节
F32 特征；单帧 2549 包、四帧 10195 包。分片头标识原字节位置，缺包按位置置零。
当前 fork 不协商 RTX；针对实际 `_retransmit` 的负例及 RTX 正控制均通过。
RTCP/NACK 保留，不另造可靠传输或隐式重传。

`5qi_7` 原版队列 1000 包，`congested` 为 1 Mbit/s / 200 包队列。
大特征突发造成的队列溢出可能远高于配置随机丢包率；RTP 捕获数、tc 丢包数与
模型可用分片数不能混用，父子 qdisc 丢包也不能简单相加。

## 应用源码补丁

补丁基于上游 [jianbing119/6G-Testbed](https://github.com/jianbing119/6G-Testbed)
的精确提交 `8581bf4974ccb35a6cc14791caebe7fcae4a0502`。请使用新目录，避免
覆盖自己的 Testbed 修改。将本结果仓库克隆到相邻目录后执行：

```bash
git clone https://github.com/jianbing119/6G-Testbed.git 6G-Testbed-minicpm
cd 6G-Testbed-minicpm
git checkout 8581bf4974ccb35a6cc14791caebe7fcae4a0502
git apply --check ../minicpm-o-2_6-streamingbench-eval/patches/minicpm_original_testbed.patch
git apply ../minicpm-o-2_6-streamingbench-eval/patches/minicpm_original_testbed.patch
```

补丁在原版 `git archive` 生成的干净副本上实际应用并运行 27 项 MiniCPM 检查和
3 项原评分回归检查；还从该副本的原版 orchestrator/server 实跑
`no_emulation` / `lossy` 各 5 题 RTC，重新编码核验全部 10 条实际特征。
[干净补丁端到端证据](../results/testbed_rtc_clean_patch_smoke_20261010/audit/report_zh.md)
仅证明补丁能运行，不能当作完整网络准确率。
运行细节见打补丁后的 `aitestbed/docs/minicpm_testbed_rtc.md`。
本结果仓库根目录的 `minicpm_split/native.py` 保留此前 Windows 离线实验版本；
Linux RTC 源码在此补丁内，勿将两种运行方式混用。

## 外部资产与运行

复用原版 Testbed 的 Linux Python 3.12 环境及系统 `tc`/`tcpdump`，
显式设置 `PYTHONPATH` 到打补丁后的 `aiortc-main-clean/src`、`aitestbed` 和
`netemu/src`。依赖环境、CUDA 驱动及 sudo 权限需在自己的机器准备。
模型/原视频/帧缓存不随本仓库上传；已有离线评测的下载及帧核验入口见
[先前复现说明](feature_packet_loss.md)。模型、运行时、manifest 和每帧均有 SHA-256。

原生运行时使用官方 b11516 CUDA 12.8 Linux x64 构建；
补丁中 `minicpm_split/native.py` 的 `LINUX_RUNTIME_SHA` 固定 C ABI 与库身份，
不接受任意版本的共享库。运行前须提供匹配的 CUDA runtime / cuBLAS 库路径。
视觉端只需从同一模型生成的无张量 tokenizer 文件：

```bash
cd aitestbed
python -m minicpm_split.assets /path/to/Model-7.6B-Q4_K_M.gguf models/minicpm-o-tokenizer-only.gguf
python -m minicpm_split.prepare_rtc --assets /path/to/mobileAI-assets \
  --runtime /path/to/llama-b11516 --output results/my_rtc
export MINICPM_RTC_CONFIG="$PWD/results/my_rtc/configs/native4_model.json"
```

`--assets` 目录布局应包含 `models/`、`data/videos/001.mp4`…`025.mp4`、
`data/native_frames/` 和 `results/q4_k_m_single_native/questions_manifest.json`。
此工具创建指向原视频的符号链接。切换单帧时使用 `single` 配置并重启自己的 server。

```bash
# 终端一；将现有 venv Python 路径保存为 PYTHON。
sudo -n env "PYTHONPATH=$PYTHONPATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
  "MINICPM_RTC_CONFIG=$MINICPM_RTC_CONFIG" "$PYTHON" \
  -m server.realtime_vlm_server --model minicpm \
  --config results/my_rtc/configs/native4_scenarios.yaml --signaling-port 18124

# 终端二；原版 CLI 保留每视频之间 20 秒间隔。
"$PYTHON" -m minicpm_split.rtc_batch --root results/my_rtc --mode native4 \
  --profiles no_emulation --runs 25 --capture-root /path/to/local/captures

# 停止上述服务器后核验，避免占用同一 GPU。
"$PYTHON" -m minicpm_split.rtc_report --root results/my_rtc --modes native4 \
  --profiles no_emulation --reencode --output results/my_rtc/audit
```

服务器仅给自己的 UDP socket 请求 32 MiB 接收缓冲（Linux 报告有效 64 MiB），
因此需 sudo。没有修改全局 sysctl 或原网络 profile 的队列、丢包、带宽和时延。
最后一帧采用显式结束和 5 秒媒体排空；晚到或重组丢弃也计入实际特征缺失。

## 独立重评分与计时

已保存的答案可在 CPU 上重评分，无需模型。进入打补丁后的 `aitestbed`：

```bash
python -m minicpm_split.rtc_report \
  --root /path/to/result-repo/results/testbed_rtc_baseline_20261010 \
  --modes single native4 --profiles no_emulation \
  --questions /path/to/result-repo/results/q4_k_m_single_native/questions_manifest.json \
  --output /path/to/fresh-audit
```

每条件分母固定 125；缺失或无效答案计错，标准 label 不传入模型推理提示词。
CPU 审计检查固定 manifest、帧索引、分片位置与保存哈希；`--reencode` 还需原资产
和原生 GPU 库，能重建实际原始/置零特征字节。部分丢包的字节重建不能由 CPU
只读分片索引替代。

`metadata_to_result_seconds` 是服务器登记特征元数据至答案生成，
不含客户端编码与回传；`server_inference_seconds` 包含串行 GPU 等待。
答案一次性返回，未测逐 token TTFT。原版 `original_testbed_report.json` 中
TTFT/TTLT/stall 等字段仍保留原名，但这里的五个事件是五道独立问题的整条答案，
不能解释为语言模型逐 token 流式时延或卡顿。

原始抓包与 SQLite 可能含短期 ICE 信令，仅本机留存。
`scripts/audit_testbed_network.py` 从它们生成不含 SDP 的包长、消息组、队列和
全局 UDP 增量摘要；全局计数并非某个 socket 的独立计数。
