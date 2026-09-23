# 把本地合并结果推成一个新分支（不动远端 main）
#
# 用途：等拿到 GitHub 写权限后，一条命令完成 —— 重建分支（保证与 master 一致）并推送。
#
# 在仓库根用 PowerShell 运行（Windows PowerShell 5.1 与 PowerShell 7 都可以）：
#
#   .\scripts\push_branch.ps1 -DryRun      # 只重建 + 校验 + 显示将要推什么，不推
#   .\scripts\push_branch.ps1              # 重建 + 推送到 origin
#   .\scripts\push_branch.ps1 -RebuildOnly
#   .\scripts\push_branch.ps1 -Remote fork # 推到 fork（先 git remote add fork <url>）
#
# 若被执行策略拦住，用：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\push_branch.ps1 -DryRun
#
# 注意：本文件是 UTF-8 **带 BOM** 保存的 —— Windows PowerShell 5.1 读无 BOM 的 UTF-8
# 会按 GBK 解码，中文注释会显示成乱码并导致语法错误。改动本文件后请保持 BOM。
#
# 设计要点
#   * 分支的**父提交固定为远端 main**，所以在 GitHub 上能直接开 PR、diff 清晰；
#   * 用 `git commit-tree` 重建提交，**从不切换分支、从不碰工作区**，
#     避免 checkout 把工作区清掉的意外；
#   * 只推这一个分支引用，绝不动 `main`。
param(
    [string]$Remote = "origin",
    [string]$Branch = "merge/netsec-protocol-v1",
    [string]$Source = "master",
    [string]$Upstream = "main",
    [switch]$DryRun,
    [switch]$RebuildOnly
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Info($m) { Write-Host "  $m" }
function Head($m) { Write-Host ""; Write-Host "=== $m ===" -ForegroundColor Cyan }

Head "0. 前置检查"
if (git status --porcelain) {
    Write-Host "  [警告] 工作区不干净，请先提交或 stash：" -ForegroundColor Yellow
    git status --short | ForEach-Object { Info $_ }
}
if (-not (git rev-parse --verify -q $Source)) { throw "找不到源分支 $Source" }
if (-not (git remote | Select-String -SimpleMatch $Remote)) { throw "找不到远端 $Remote（先 git remote add $Remote <url>）" }
Info "仓库      : $repo"
Info "源分支    : $Source = $(git rev-parse --short $Source)"
Info "远端      : $Remote"
try {
    git fetch -q $Remote $Upstream
    Info "fetch $Remote/$Upstream 成功"
} catch {
    Write-Host "  [警告] fetch 失败（网络？），改用本地缓存的 $Remote/$Upstream" -ForegroundColor Yellow
}
$parent = git rev-parse "$Remote/$Upstream"

Head "1. 重建分支（commit-tree，不碰工作区）"
$tree = git rev-parse "$Source^{tree}"
$msgPath = Join-Path $repo "_cmsg.tmp"
$message = @"
merge: 四方向合并为统一仓库（src/ 布局）+ 评测协议 v1

把 chethuhn / network / safenetwork / ae_repro 四个平行目录合并成一个仓库，
并落实统一评测协议 v1。原四个本地目录保持不动。

布局：agentenvs|algorithms|utils -> src/ 下同名包；顶层脚本 -> scripts/；
      平台 README -> docs/README_platform.md；docs/requirements.txt -> docs/requirements_platform.txt
新增：PROTOCOL.md（唯一评测口径）、src/ae_repro、src/icps_detection、
      ae_paper 检测器注册、scripts/eval_all_detectors.py、tests/、docs/{DECISIONS,ACCEPTANCE}.md

协议 v1 的两个关键修复：
  1) 规范数据包评估集排除 Monday（检测器拟合数据）：
     自评重复 14.02% (4066/29000) -> 0.70% (176/25000)
  2) 主口径 = 良性 95% 分位阈值（fit 只喂良性、不传 y_test），
     F1 扫描只作乐观上界；同一 IF 在两种口径下 F1 = 0.591 vs 0.465

主口径 F1(ROC-AUC)：ae_paper 0.6521(0.8474) > autoencoder 0.6412(0.8021)
  > kalman_frozen 0.5797(0.7841) > kalman_decay 0.5764(0.7795)
  > if 0.4560(0.7327) > null 0.4103(0.5000)

验收：verify_pipeline 全通过 / templates 自检通过 / 契约自检 35-35 / pytest 10 passed

与 main 的分叉：main 的 autoencoder_detector.py 与 maddpg.py 已逐字节备份到
legacy/diverged-from-remote/（含 3 种恢复方式），需作者确认 embed_dim 一行以谁为准。

详见 docs/PR_DESCRIPTION.md、PROTOCOL.md、docs/DECISIONS.md、docs/ACCEPTANCE.md
"@
[System.IO.File]::WriteAllText($msgPath, $message, [System.Text.UTF8Encoding]::new($false))
$new = git commit-tree $tree -p $parent -F $msgPath
Remove-Item $msgPath -Force
git branch -f $Branch $new | Out-Null
Info "分支 $Branch = $(git rev-parse --short $new)"
Info "父提交        = $(git rev-parse --short $new^)   (=$Remote/$Upstream)"
Info "树与 $Source 一致 : $((git rev-parse "$Branch^{tree}") -eq $tree)"
Info "文件数        = $((git ls-tree -r --name-only $Branch | Measure-Object).Count)"

Head "2. 将要推送的内容"
git --no-pager diff --shortstat "$Remote/$Upstream" $Branch
git --no-pager log --oneline "$Remote/$Upstream"..$Branch

if ($RebuildOnly) { Head "完成（-RebuildOnly，未推送）"; exit 0 }

Head "3. 推送"
if ($DryRun) {
    Info "[-DryRun] 将执行：git push -u $Remote $Branch"
    git push --dry-run -u $Remote $Branch
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [提示] 与远端的 dry-run 校验没成功（多半是网络问题）。" -ForegroundColor Yellow
        Write-Host "         本地分支已经重建完成，上面第 2 节的 diff 就是将要推送的内容。" -ForegroundColor Yellow
    }
    Head "完成（-DryRun，未真正推送）"
    exit 0
}
git push -u $Remote $Branch
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host ""
    Write-Host "  推送失败（exit $code）。若是 403 / Permission denied，说明当前凭据的账号对该仓库没有写权限：" -ForegroundColor Yellow
    Write-Host "    ① 让仓库 owner 把该账号加成 collaborator，然后重跑本脚本；"
    Write-Host "    ② cmdkey /delete:LegacyGeneric:target=git:https://github.com  之后改用 owner 账号认证；"
    Write-Host "    ③ 网页 Fork 到自己的账号，然后 git remote add fork <fork-url> 并 pwsh -File scripts\push_branch.ps1 -Remote fork。"
    exit $code
}
Head "推送成功"
Info "分支已推送到 $Remote/$Branch；远端 main 未被改动。"
Info "开 PR：https://github.com/ZhongRuiYu-mine/NetworkSecurity/compare/main...$Branch"
