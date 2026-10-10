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

## 2026-10-10：补测 20%–50% 区间

新增 25%、30%、35%、40%、45%、50% 六个点，继续使用同一模型、运行时、
125 道固定问题、原生帧和种子。每个新增条件均完整推理；没有按期望曲线筛选题目或重跑种子。
主图固定展示 0%–50%，另有完整范围图保留 100% 全零对照；标记为实测值，连线不代表拟合。

```powershell
# 仍需按上述流程启动语言模型服务器。输出目录可用于相同配置的断点续跑。
python -m minicpm_split.packet_loss --runtime runtime/vulkan --projector models/mmproj-model-f16.gguf --questions results/q4_k_m_single_native/questions_manifest.json --frames data/native_frames --output results/feature_packet_loss_midrange_rerun --loss-rates 0.25 0.30 0.35 0.40 0.45 0.50

# 合并旧实验、补测与全零对照，逐题重评分及核验后画图。
python -m minicpm_split.packet_loss_curve --sources results/feature_packet_loss_rerun results/feature_packet_loss_midrange_rerun results/feature_packet_loss_allzero_rerun --questions results/q4_k_m_single_native/questions_manifest.json --text-control results/zero_feature_controls/summary.json --output results/feature_packet_loss_curve_rerun

# 仅复核仓库中已发布的 3250 条回答，无需模型或视频。
python -m minicpm_split.packet_loss_curve --sources results/feature_packet_loss results/feature_packet_loss_midrange results/feature_packet_loss_allzero --questions results/q4_k_m_single_native/questions_manifest.json --text-control results/zero_feature_controls/summary.json --output results/feature_packet_loss_curve_reaudit
```

合并脚本拒绝重复条件、不同模型/提示词/种子/输入设定以及不完整问题集合，
并核对新增运行与原运行每题的原始特征哈希、帧元数据完全一致。
各条件只使用一组随机掩码，准确率仍以 125 题为分母；纯文本控制不计入 3250 条丢包实验回答。
报告及 PNG/PDF/SVG 见 `results/feature_packet_loss_curve/`。


## 2026-10-10：补测 50%–100% 区间

新增 55%、60%、65%、70%、75%、80%、85%、90%、95%、99% 十个点，
单帧与连续四帧各 125 题，共新增 2500 次推理。沿用同一模型、问题、原生帧、提示词及种子。
合并既有结果后为 23 个丢包率、46 个完整条件、5750 次回答。
图中保留该区间的所有点及 100% 全零对照，另提供 0%–100% 全范围图。

```powershell
# 先启动上文的语言模型服务器；此命令执行新增十点的实际推理。
python -m minicpm_split.packet_loss --runtime runtime/vulkan --projector models/mmproj-model-f16.gguf --questions results/q4_k_m_single_native/questions_manifest.json --frames data/native_frames --output results/feature_packet_loss_highrange_rerun --loss-rates 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99

# 仅复核全部已发布的 5750 条回答并重新画图，无需权重或视频。
python -m minicpm_split.packet_loss_highrange_curve --sources results/feature_packet_loss results/feature_packet_loss_midrange results/feature_packet_loss_highrange results/feature_packet_loss_allzero --questions results/q4_k_m_single_native/questions_manifest.json --text-control results/zero_feature_controls/summary.json --output results/feature_packet_loss_highrange_curve_reaudit
```

实际复现推理后，将上述合并命令的各源目录替换为对应 `_rerun` 目录。
脚本逐题重评分、重建掩码并检查服务器特征哈希及跨运行输入一致性；不修改模型或筛选结果。
相邻点变化表同时报告下降百分点和按区间宽度归一化的差商，后者不代表精确拐点。
本轮仍是离线张量擦除；正式原版 Testbed RTC/network profile 评测另行报告。
报告及图表见 `results/feature_packet_loss_highrange_curve/`。
