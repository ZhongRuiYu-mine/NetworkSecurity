# 把本地合并结果推成一个分支（默认只推分支，不动远端 main）
#
# 用途：把本地 master 的当前内容重建成一个提交并推送。
#
# ⚠️ 2026-10-08 起远端 main 已被快进到合并后的树，不再是"合并前快照"。
#    本脚本默认仍只推 $Branch；要同步 main 请显式用 -AlsoUpdateMain。
#
# 在仓库根用 PowerShell 运行（Windows PowerShell 5.1 与 PowerShell 7 都可以）：
#
#   .\scripts\push_branch.ps1 -DryRun      # 只重建 + 校验 + 显示将要推什么，不推
#   .\scripts\push_branch.ps1              # 重建 + 推送到 origin
#   .\scripts\push_branch.ps1 -RebuildOnly
#   .\scripts\push_branch.ps1 -Remote fork # 推到 fork（先 git remote add fork <url>）
#   .\scripts\push_branch.ps1 -AlsoUpdateMain   # 额外把 main 快进到同一棵树（非 force）
#
# 若被执行策略拦住，用：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\push_branch.ps1 -DryRun
#
# 注意：本文件是 UTF-8 **带 BOM** 保存的 —— Windows PowerShell 5.1 读无 BOM 的 UTF-8
# 会按 GBK 解码，中文注释会显示成乱码并导致语法错误。改动本文件后请保持 BOM。
#
# 设计要点
#   * 分支的**父提交固定为远端 $Upstream**，所以在 GitHub 上能直接开 PR、diff 清晰；
#   * 用 `git commit-tree` 重建提交，**从不切换分支、从不碰工作区**，
#     避免 checkout 把工作区清掉的意外；
#   * 默认只推这一个分支引用；只有显式 -AlsoUpdateMain 才会推 $Upstream。
param(
    [string]$Remote = "origin",
    [string]$Branch = "merge/netsec-protocol-v1",
    [string]$Source = "master",
    [string]$Upstream = "main",
    [switch]$DryRun,
    [switch]$RebuildOnly,
    [switch]$AlsoUpdateMain
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

Head "0b. 是否还有事情可做"
$srcTree = git rev-parse "$Source^{tree}"
$upTree = git rev-parse "$Remote/$Upstream^{tree}"
$brTree = try { git rev-parse "$Remote/$Branch^{tree}" } catch { "(无此分支)" }
Info "master 树        : $($srcTree.Substring(0,12))"
Info "$Remote/$Upstream 树 : $($upTree.Substring(0,12))"
Info "$Remote/$Branch 树 : $(if ($brTree.Length -ge 12) { $brTree.Substring(0,12) } else { $brTree })"
if ($srcTree -eq $upTree) {
    # 远端上游已经和本地 master 同一棵树 —— 再"重建"只会制造一个内容相同的空提交。
    # 2026-10-08 起 main 已被快进，所以这是**正常**状态，不是错误。
    Head "完成（远端已是同一棵树，无需推送）"
    Info "远端 $Remote/$Upstream 的内容已与本地 $Source 完全一致。"
    if ($AlsoUpdateMain) { Info "-AlsoUpdateMain 已指定，但 $Upstream 已经是目标内容，无需再推。" }
    Info "要改内容请先在本地提交，然后重跑本脚本。"
    exit 0
}

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
    if ($AlsoUpdateMain) { Info "[-DryRun] 并执行：git push $Remote ${Branch}:$Upstream" }
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
Info "分支已推送到 $Remote/$Branch。"
Info "开 PR：https://github.com/ZhongRuiYu-mine/NetworkSecurity/compare/$Upstream...$Branch"

if ($AlsoUpdateMain) {
    Head "4. 把 $Upstream 快进到同一棵树（非 force）"
    git push $Remote "${Branch}:$Upstream"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [失败] 推送 $Upstream 失败（exit $LASTEXITCODE）。" -ForegroundColor Yellow
        Write-Host "         若提示 non-fast-forward，说明远端 $Upstream 有本地没有的提交，" -ForegroundColor Yellow
        Write-Host "         请先 fetch+合并，不要用 --force。" -ForegroundColor Yellow
        exit $LASTEXITCODE
    }
    Info "$Upstream 已快进到与 $Branch 同一棵树。"
} else {
    Info "（未指定 -AlsoUpdateMain，远端 $Upstream 未被改动。）"
}
