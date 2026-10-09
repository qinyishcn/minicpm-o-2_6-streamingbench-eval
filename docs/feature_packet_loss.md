# 复现视觉特征分包丢包置零实验

此实验是用户在正式 Testbed 网络评测前要求的独立小实验。
只在视觉编码器与语言模型之间，对 MTU 大小分片对应的特征进行擦除并置零。
它不替代原版 Testbed 的 RTC、网卡模拟、network profile 或抓包流程。

环境沿用本仓库已完成的评测：Windows x64、Python 3.12、RTX 3070 8 GB、
作者 Q4_K_M 语言权重、F16 视觉权重、llama.cpp b11516 Vulkan。
关键模型、tokenizer 与 DLL 在加载时验证 SHA-256，其他原生 ABI 版本不保证兼容。

先按 README 下载模型和固定版本的运行时，准备 25 个视频及 native_frames.py 的缓存。
模型、视频、图像、视觉特征张量和运行时二进制不在仓库分发。

```powershell
python -m pip install -r requirements.txt
python -m minicpm_split.assets models/Model-7.6B-Q4_K_M.gguf models/minicpm-o-tokenizer-only.gguf

# 终端 1：服务器只加载语言模型
python -m minicpm_split.server --runtime runtime/vulkan --model models/Model-7.6B-Q4_K_M.gguf

# 终端 2：端侧只加载视觉模型；此目录必须是新目录或相同配置的断点续跑目录
python -m minicpm_split.packet_loss --runtime runtime/vulkan --projector models/mmproj-model-f16.gguf --questions results/q4_k_m_single_native/questions_manifest.json --frames data/native_frames --output results/feature_packet_loss_rerun

# 100% 特征置零对照
python -m minicpm_split.packet_loss --runtime runtime/vulkan --projector models/mmproj-model-f16.gguf --questions results/q4_k_m_single_native/questions_manifest.json --frames data/native_frames --output results/feature_packet_loss_allzero_rerun --loss-rates 1

# 从已保存回答、原始标签和掩码独立评分并生成报告，无需再次推理
python -m minicpm_split.packet_loss_report --results results/feature_packet_loss_rerun --allzero-control results/feature_packet_loss_allzero_rerun --questions results/q4_k_m_single_native/questions_manifest.json
python -m unittest discover -s tests -p 'test_minicpm*.py' -v
```

主实验默认 0%、0.1%、1%、5%、10%、20% 独立 Bernoulli 丢包，基础种子 20261009。
每个输入方式、每道题派生独立种子；不同丢包率使用相同随机序列，形成嵌套掩码。
每题只有一个随机掩码，这是一轮探索性实验。

1500 字节为 IP MTU，假定 IPv4/UDP/RTP/特征头/SRTP 标签分别占 20/8/12/16/10 字节。
对齐 float32 后，每个完整片段为 1432 字节特征，包长最多 1498 字节。
元数据、问题和文本 token 不擦除；丢失分片对应的特征全部置零，形状和位置保持不变。
本地 TCP 仅负责把已擦除的张量完整送给语言模型，不模拟丢包传输效率。

报告入口为 results/feature_packet_loss/report_zh.md。逐题记录保存原始回答、标签、随机种子、
丢包位置、置零字节数、原始/受损特征哈希及服务器收包哈希。
0% 特征置零实验另与之前的无丢包拆分基线核对答案和首步 logits，证据保存在 audit.json。

只复核仓库内已发布结果时，无需下载权重、运行时或视频：

```powershell
python -m pip install -r requirements.txt
python -m minicpm_split.packet_loss_report --results results/feature_packet_loss --allzero-control results/feature_packet_loss_allzero --questions results/q4_k_m_single_native/questions_manifest.json --split-baseline results/feature_split_baseline
```

这会独立重评分 1750 条保存的回答，重建每个丢包掩码并核对接收哈希，
同时核对 250 个无丢包样例的特征、答案和首步完整 logits 哈希。
该操作会重新生成报告及图表；图表和 PDF 的文件哈希可能受本地绘图版本与生成时间影响。
