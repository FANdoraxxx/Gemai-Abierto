# MERT-v1-330M（第三方模型，随本仓库分发）

- 来源：https://huggingface.co/m-a-p/MERT-v1-330M ，固定版本 `5240c2708a5acaee1007f43fb9735c7dcd0b78c9`
- 作者：Yizhi Li, Ruibin Yuan, Ge Zhang 等（Multimodal Art Projection, m-a-p）
- 许可：**CC BY-NC 4.0**（https://creativecommons.org/licenses/by-nc/4.0/ ）。仅限非商业用途，使用时需署名。
- 修改说明：原始权重 `pytorch_model.bin` 被转存为 15 个 safetensors 分片（`model-000xx-of-00015.safetensors` + `model.safetensors.index.json`），
  以满足 GitHub 单文件 100 MB 的限制。转存前后所有 403 个张量逐位相同，第 8 层输出逐位相同（max abs diff = 0）。
  `config.json`、`configuration_MERT.py`、`modeling_MERT.py`、`preprocessor_config.json` 为原文件，未修改。
- 本项目只把 MERT 作为冻结的特征提取器使用（第 8 层，1024 维）。

引用：

```
@misc{li2023mert,
  title={MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training},
  author={Yizhi Li and Ruibin Yuan and Ge Zhang and others},
  year={2023},
  eprint={2306.00107},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```
