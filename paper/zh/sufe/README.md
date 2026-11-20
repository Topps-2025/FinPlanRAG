# 上财本科毕业论文中文稿

- 源文件：`main.tex`
- 参考文献：`references.bib`
- 图片：`figures/`
- PDF：`../FinPlanRAG_Chinese_SUFE_2026.pdf`

编译：

```text
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

版式依据上海财经大学 2024 年校级本科毕业论文规定：A4，左边距 3cm，
右边距 2cm，上下边距 2.5cm；正文五号宋体；章标题三号黑体，节标题四号黑体，
小节五号黑体；中文摘要四号宋体；英文摘要五号 Arial；GB/T 7714-2015
数字制参考文献。封面中的学号、指导教师为空白线，提交前替换。

当前 PDF 共 40 页；1,005 条基准发布在 `../../data/`，复现命令见
`../../reproduce.md`。
