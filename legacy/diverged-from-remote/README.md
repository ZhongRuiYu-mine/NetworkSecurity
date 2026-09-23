# 与线上仓库分叉的文件（备份 + 恢复说明）

> 这个目录**只是备份和说明**，不参与任何代码路径（`legacy/` 不被打包、不被 import）。

## 为什么有这个目录

合并时逐文件比对线上仓库 `ZhongRuiYu-mine/NetworkSecurity` 与本仓库，
24 个文件里有 **2 个在合并前就已经分叉**，两边内容不同：

| 文件 | 线上 `main`（2026-09-17） | `safenetwork` 快照（本分支采纳，2026-09-21） |
|---|---|---|
| `agentenvs/detectors/autoencoder_detector.py` | `self.embed_dim = 0` | `self.embed_dim = int(embed_dim) if embed_dim is not None else 0` |
| `algorithms/maddpg.py` | 无调试输出 | 每 100 次更新打印一行 `[DEBUG] q_det …` |

这两个文件是那部分代码的开发者自己改的，**为避免分叉版本被覆盖后找不回来**，
把线上版本原样备份在此：

```
legacy/diverged-from-remote/
├── README.md                          ← 本文件
├── autoencoder_detector.remote.py     ← 线上 main 的版本，逐字节一致（sha256 5b37bbbdd6bd0cc1…）
└── maddpg.remote.py                   ← 线上 main 的版本，逐字节一致（sha256 92b34343c6e14a9a…）
```

## 线上版本与本分支版本的差异（完整，只有这两处）

### 1. `autoencoder_detector.py`（第 62 行附近）

```diff
--- 线上 main
+++ 本分支（来自 safenetwork 快照）
@@ -62,3 +62,3 @@
         self._sd = 1.0
-        self.embed_dim = 0
+        self.embed_dim = int(embed_dim) if embed_dim is not None else 0
```

**影响**：线上写法把 `embed_dim` 硬编码成 `0`，会忽略构造时传入的 `embed_dim`。
而 `INTERFACE.md` §2.4 要求"环境把统一的 `detector_embed_dim` 应用到检测器上"，
所以线上版本在"四种机制统一 `embed_dim=4`"的对比实验里会让 AE 的信号维度掉到 3，
观测维度与其他机制不一致 → `set_detector()` 热替换会抛 `ValueError`。
`tests/test_smoke.py::test_registry_has_all_mechanisms` 会拦住这种情况
（本分支实测四种机制 `signal_dim` 全为 7）。

> ⚠️ 这只是从代码行为推断的结论。**如果线上那版是有意为之**（例如显式让 AE 不输出
> embedding），请把这一行按线上改回去，并同步修改协议里的 `embed_dim` 约定。

### 2. `algorithms/maddpg.py`（第 302 行附近）

```diff
--- 线上 main
+++ 本分支（来自 safenetwork 快照）
@@ -302,2 +302,7 @@
                     loss_class = -(lp_taken * adv).mean() * cfg.class_lr_scale
+                    if self.total_updates % 100 == 0 and i == 0:
+                        print(f"[DEBUG] q_det mean={q_det.mean().item():.4f}, std={q_det.std().item():.4f}, "
+                              f"adv mean={adv.mean().item():.4f}, std={adv.std().item():.4f}, "
+                              f"lp_taken mean={lp_taken.mean().item():.4f}, "
+                              f"logits std={cur_logits[i].std().item():.4f}")
                     if cfg.entropy_coef > 0:
```

**影响**：纯调试输出，无功能影响（训练时会多打印若干行）。
如果觉得吵，直接删掉这 5 行即可。

## 怎么恢复线上版本

### 方式 A：不碰 git，直接拷文件（推荐给不熟 git 的同学）

把备份文件覆盖到新分支的对应位置（注意目标路径变了，`agentenvs/` → `src/agentenvs/`）：

```powershell
Copy-Item legacy\diverged-from-remote\autoencoder_detector.remote.py `
          src\agentenvs\detectors\autoencoder_detector.py -Force
Copy-Item legacy\diverged-from-remote\maddpg.remote.py `
          src\algorithms\maddpg.py -Force
```

### 方式 B：用 git 从线上 `main` 直接取

```powershell
# 看差异
git diff origin/main:agentenvs/detectors/autoencoder_detector.py -- src/agentenvs/detectors/autoencoder_detector.py

# 把线上版本取出来（路径改了，所以要手动落到 src/ 下）
git show origin/main:agentenvs/detectors/autoencoder_detector.py > src/agentenvs/detectors/autoencoder_detector.py
git show origin/main:algorithms/maddpg.py                       > src/algorithms/maddpg.py
```

### 方式 C：只要回看线上原文件（什么都不改）

线上仓库的 `main` 分支**没有被本次合并改动**（我们的提交只进新分支），
所以任何时候都能直接访问：

- <https://github.com/ZhongRuiYu-mine/NetworkSecurity/blob/main/agentenvs/detectors/autoencoder_detector.py>
- <https://github.com/ZhongRuiYu-mine/NetworkSecurity/blob/main/algorithms/maddpg.py>

## 建议

1. 请这两个文件的作者确认一下 **`embed_dim` 那一行哪个版本为准**（我倾向本分支的写法，理由见上）。
2. 若线上版本为准，在新分支上提一个 commit 改回去，别直接改 `main`。
3. 其余 22 个文件不需要担心：13 个逐字节一致（只差 Windows checkout 的换行符），
   9 个的差异全部来自本次布局调整（详见 `docs/DECISIONS.md` 第 5 节）。
