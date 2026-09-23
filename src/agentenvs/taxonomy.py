"""攻击类别表 + 智能体分工（specialization）配置。

CICIDS2017 原始 Label 列里带 BOM / 破损编码（例如 `Web Attack \ufffd Brute Force`），
这里统一归一化成干净的类名，并按论文的设定（每个智能体专注一类威胁）
把类别分配给多个智能体。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

#: 环境默认的威胁类别（对齐论文里的 DDoS / Phishing / Spoofing 等描述，
#: 落到 CICIDS2017 上就是下面这几类）。
#:
#: 注意最后一项 OTHER_CLASS 是"兜底桶"：真实数据里出现未列入清单的攻击标签时
#: 会被映射到它，而不是被误当成 BENIGN。前 5 类正好喂给 5 个专注型智能体，
#: 兜底桶由所有智能体共同负责（不存在"专门抓未知攻击"的智能体）。
OTHER_CLASS = "Other"

DEFAULT_ATTACK_CLASSES: List[str] = [
    "DDoS",          # Friday Afternoon DDos
    "PortScan",      # Friday Afternoon PortScan
    "Bot",           # Friday Morning (Botnet)
    "DoS",           # Wednesday: Hulk / GoldenEye / slowloris / Slowhttptest
    "WebAttack",     # Thursday Morning: Brute Force / XSS / Sql Injection
    OTHER_CLASS,     # 兜底：Heartbleed / Infiltration / Patator 等
]

#: 折叠成单一"攻击类"的两种可选清单
BINARY_ATTACK_CLASSES: List[str] = ["Attack"]

#: 把主要攻击族拆开（不含兜底桶，未覆盖的仍归 OTHER）
FULL_ATTACK_CLASSES: List[str] = [
    "DDoS", "DoS", "PortScan", "Bot", "WebAttack",
    "FTP-Patator", "SSH-Patator", "Heartbleed", "Infiltration",
    OTHER_CLASS,
]

BENIGN = "BENIGN"

#: 原始标签 -> 归一类名
_LABEL_ALIASES: Dict[str, str] = {
    "benign": BENIGN,
    "ddos": "DDoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "heartbleed": "Heartbleed",
    "portscan": "PortScan",
    "bot": "Bot",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "infiltration": "Infiltration",
    "web attack - brute force": "WebAttack",
    "web attack - xss": "WebAttack",
    "web attack - sql injection": "WebAttack",
}

#: 面向 `WebAttack \ufffd Brute Force` 这类破损标签的正则
_WEB_ATTACK_RE = re.compile(r"web\s*attack", re.IGNORECASE)
_DOS_RE = re.compile(r"^dos[\s\-_]", re.IGNORECASE)


def _clean(raw: str) -> str:
    """去掉 BOM / 非 ASCII 噪声字符，压缩空白。"""
    s = str(raw)
    s = s.replace("\ufeff", "").replace("\ufffd", "-").replace("ï¿½", "-")
    s = "".join(ch if (ch.isascii() and ch.isprintable()) else "-" for ch in s)
    s = re.sub(r"[\s\-_]*\-[\s\-_]*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def normalize_label(raw: str) -> str:
    """把 CICIDS2017 的原始标签字符串映射到规范类名。"""
    s = _clean(raw)
    key = s.lower()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    if _WEB_ATTACK_RE.search(key):
        return "WebAttack"
    if _DOS_RE.match(key) or key.startswith("dos"):
        return "DoS"
    if key.startswith("ddos"):
        return "DDoS"
    if "patator" in key:
        return "FTP-Patator" if key.startswith("ftp") else "SSH-Patator"
    # 未收录的攻击名：直接用清理后的原名，保证可扩展
    return s.replace(" ", "") or "UNKNOWN"


@dataclass
class ClassSpec:
    """环境使用的类别空间。"""

    benign_name: str = BENIGN
    attack_classes: List[str] = field(default_factory=lambda: list(DEFAULT_ATTACK_CLASSES))

    @property
    def names(self) -> List[str]:
        """第 0 类固定为 BENIGN，其余为攻击类。"""
        return [self.benign_name] + list(self.attack_classes)

    @property
    def num_classes(self) -> int:
        return 1 + len(self.attack_classes)

    @property
    def num_attack_classes(self) -> int:
        return len(self.attack_classes)

    def index_of(self, canonical: str) -> int:
        """
        规范类名 -> 索引。

        映射规则：
          * BENIGN -> 0
          * 在 attack_classes 里 -> 对应索引
          * 未列举的其他攻击 -> OTHER_CLASS 的位置（若清单里没有 OTHER，
            则退回最后一个攻击类，仍然**不会**变成 BENIGN）
        """
        if canonical == self.benign_name:
            return 0
        try:
            return 1 + self.attack_classes.index(canonical)
        except ValueError:
            pass
        if not self.attack_classes:
            return 0
        if OTHER_CLASS in self.attack_classes:
            return 1 + self.attack_classes.index(OTHER_CLASS)
        return len(self.attack_classes)  # 最后一个攻击类当作"其它"

    def name_of(self, idx: int) -> str:
        names = self.names
        return names[idx] if 0 <= idx < len(names) else "UNKNOWN"

    @classmethod
    def from_preset(cls, preset: str) -> "ClassSpec":
        preset = (preset or "default").lower()
        if preset in ("default", "5"):
            return cls(attack_classes=list(DEFAULT_ATTACK_CLASSES))
        if preset in ("full", "all", "9", "10"):
            return cls(attack_classes=list(FULL_ATTACK_CLASSES))
        if preset in ("binary", "1"):
            return cls(attack_classes=list(BINARY_ATTACK_CLASSES))
        raise ValueError(f"未知类别预设 {preset!r}（可选：default / full / binary）")


def named_attack_indices(spec: ClassSpec) -> List[int]:
    """返回"可被专门指派"的攻击类索引（排除 BENIGN 和兜底桶 Other）。"""
    idx = [
        1 + i
        for i, n in enumerate(spec.attack_classes)
        if n not in (OTHER_CLASS, BENIGN)
    ]
    return idx or [1] * max(spec.num_attack_classes, 1)


def default_agents(num_agents: int, spec: ClassSpec) -> List[Dict]:
    """
    生成智能体分工：每个智能体"专注"一类攻击（论文：DDoS / phishing / spoofing ...）。

    返回 list[dict]，每项:
        {"id": i, "name": "agent_DDoS", "focus": 类别索引}

    超出可指派类别数的智能体轮流复用（轮流制），保证分工表长度恒为 num_agents。
    """
    pool = named_attack_indices(spec)
    agents: List[Dict] = []
    for i in range(num_agents):
        cls_idx = pool[i % len(pool)]
        agents.append(
            {
                "id": i,
                "name": f"agent_{spec.name_of(cls_idx)}"
                if i < len(pool)
                else f"agent_{spec.name_of(cls_idx)}_{i // len(pool) + 1}",
                "focus": cls_idx,
            }
        )
    return agents


def resolve_specialization(agents: Optional[Sequence], num_agents: int, spec: ClassSpec) -> List[Dict]:
    """把用户传入的分工配置补齐成 num_agents 条。"""
    if not agents:
        return default_agents(num_agents, spec)
    out: List[Dict] = []
    for i in range(num_agents):
        if i < len(agents):
            a = dict(agents[i])
            a.setdefault("id", i)
            a.setdefault("focus", 1 + (i % max(spec.num_attack_classes, 1)))
            a.setdefault("name", f"agent_{i}")
        else:
            a = default_agents(num_agents, spec)[i]
        # focus 允许是类名或索引
        f = a["focus"]
        if isinstance(f, str):
            a["focus"] = spec.index_of(normalize_label(f))
        out.append(a)
    return out
