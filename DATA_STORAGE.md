# FinPlanRAG 外置数据说明

本 GitHub 工作目录 `C:\Users\Lenovo\Desktop\Paper\FinPlanRAG` 只保留研究框架、核心代码、测试和冻结协议。原始数据、构建语料、模型权重、实验结果与历史归档统一存放在：

`D:\Engineering\FinPlanRAG\database`

目录约定：

- `preexperiments/data/`：原始公开数据、PDF 与构建后的实验语料；
- `preexperiments/models/`：本地模型权重；
- `preexperiments/results/`：实验运行结果；
- `tmp/`：论文、镜像索引和临时下载；
- `archive/`：旧版草稿和生成缓存；
- `storage_manifest_v1.json`：外置资产大小与 SHA-256 清单。

代码默认读取上述路径。迁移到其他机器时设置环境变量：

```powershell
$env:FINPLANRAG_STORAGE_ROOT = 'E:\YourPath\FinPlanRAG\database'
```

不要把外置数据库直接提交到 GitHub。公开数据的来源、许可、下载失败和镜像访问情况仍由各实验审计文件记录。
