"""S44 王者荣耀通识参考库。

来源为用户提供的个人总结《S44王者荣耀通识》。它适合统一标注术语、解释
录像语境，但不是官方规则。角色、装备和英雄榜会随阵容、补丁与统计口径变化，
因此不得用作硬性合法性校验，也不得直接充当模型标签或训练真值。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


META = {
    "season": "S44",
    "label": "S44 王者荣耀通识参考库",
    "document": "S44王者荣耀通识.docx",
    "providedAt": "2026-09-09",
    "provenance": "user-provided-personal-summary",
    "externalVerified": False,
    "official": False,
    "defaultConfidence": 0.4,
    "usagePolicy": "术语可用于人工标注归一化；其余内容仅作解释，不得作为硬规则、模型标签或训练真值。",
}


STATUS_LABELS = {
    "definition_reference": "术语参考",
    "strategy_heuristic": "战术经验",
    "strong_cue": "强提示，非硬规则",
    "ambiguous": "表述有歧义",
    "overgeneralized": "绝对说法已降级",
    "volatile_snapshot": "版本快照",
}


def _entry(
    entry_id: str,
    category: str,
    title: str,
    summary: str,
    pages: list[int],
    status: str,
    *,
    aliases: list[str] | None = None,
    note: str = "",
    captured_at: str | None = None,
) -> dict[str, Any]:
    if status not in STATUS_LABELS:
        raise ValueError(f"未知通识条目状态：{status}")
    return {
        "id": entry_id,
        "category": category,
        "title": title,
        "summary": summary,
        "sourcePages": pages,
        "status": status,
        "statusLabel": STATUS_LABELS[status],
        "aliases": list(aliases or []),
        "capturedAt": captured_at,
        "annotationAliasAllowed": status == "definition_reference",
        "validationAllowed": False,
        "modelFeatureAllowed": False,
        "confidence": META["defaultConfidence"] if status == "definition_reference" else 0.0,
        "note": note,
    }


ENTRIES = [
    _entry("neutral_resources", "基础术语", "中立资源",
           "没有蓝红阵营归属、双方都可争夺的资源；文档举例为龙、空间之灵和红隼。",
           [1], "definition_reference", aliases=["中立资源"]),
    _entry("jungle_farming", "基础术语", "刷野",
           "英雄击杀野怪以获取经验和金币。", [1], "definition_reference",
           aliases=["刷野", "清野"]),
    _entry("jungle_invade", "基础术语", "反野",
           "进入敌方野区争夺或击杀敌方野怪。", [1], "definition_reference",
           aliases=["反野", "入侵野区", "野区入侵"]),
    _entry("buff_start", "基础术语", "红开与蓝开",
           "打野从猩红石像一侧起手称红开，从蔚蓝石像一侧起手称蓝开。",
           [1], "definition_reference", aliases=["红开", "蓝开"]),
    _entry("buff_guard", "基础术语", "看红与看蓝",
           "队友在开局或可能被入侵时保护己方红、蓝BUFF区域。",
           [1], "definition_reference", aliases=["看红", "看蓝", "守红", "守蓝"]),
    _entry("cut_carry", "基础术语", "切C",
           "优先接近并攻击敌方核心输出位，常见目标是射手或主要法术输出。",
           [1], "definition_reference", aliases=["切C", "切后排", "切双C"]),
    _entry("split_41", "战术术语", "四一分带",
           "四人抱团牵制，一人处理或推进边线，以兵线压力换取空间。",
           [1], "strategy_heuristic", aliases=["四一", "四一分带", "41分带"],
           note="单带者常见于打野或对抗路，但不能只凭英雄分路认定。"),
    _entry("split_131", "战术术语", "一三一分线",
           "三人控制中路，两名边线英雄分别处理侧线，并保持支援可能。",
           [1], "strategy_heuristic", aliases=["一三一", "131", "打三线"],
           note="文档将“打三线”用于这一语境；实际解说用词可能更宽泛。"),
    _entry("lane_swap", "战术术语", "换线",
           "英雄在对局中离开通常分路，与队友交换线路以改变对位或资源分配。",
           [1], "definition_reference", aliases=["换线", "转线"]),
    _entry("unusual_draft", "分路判断", "非常规阵容与换线分开判断",
           "阵容选择非常规英雄不等于发生换线；应以实际线路、停留时间和资源获取判断。",
           [1], "strong_cue", note="这是录像标注的重要防误判原则。"),
    _entry("lane_orientation", "分路判断", "地图半区与分路方向",
           "文档用红蓝BUFF所在半区描述对抗路和发育路方向。",
           [1], "ambiguous",
           note="镜像视角、观战UI和地图版本可能改变画面方向，不能直接用屏幕左右作为分路标签。"),
    _entry("jungler_role", "角色判断", "打野位识别",
           "惩击、打野装备、早期野区资源路线和控龙行为共同构成打野位强提示。",
           [1, 4], "overgeneralized",
           note="原文写成“必须携带惩击和打野装备”；系统改为多证据判断，避免特殊阵容误判。"),
    _entry("clash_role", "角色判断", "对抗路识别",
           "对抗路线位置、空间之灵、地图分路标识及早期单人吃线可作为综合判断依据。",
           [1], "strong_cue", note="战士或坦克只是常见类型，不是充分条件。"),
    _entry("mid_role", "角色判断", "中路识别",
           "中路线活动、兵线资源和支援路线是主要依据；法师是常见选择但不是必要条件。",
           [1], "overgeneralized",
           note="原文“中路必须是法师”不适合硬编码，已明确降级。"),
    _entry("farm_role", "角色判断", "发育路识别",
           "发育路线位置、红隼区域和早期吃线可作为判断依据；射手是常见选择。",
           [1], "strong_cue", note="英雄职业标签不能单独决定实际分路。"),
    _entry("roam_role", "角色判断", "游走位识别",
           "辅助装备、低固定吃线率、跨线支援和视野活动共同构成游走位提示。",
           [1, 3], "overgeneralized",
           note="原文写成“必须购买学识宝石”；系统不把单件装备当作充分或必要条件。"),
    _entry("hero_role_rank_snapshot", "版本资料", "英雄分路与赛事热度截图",
           "文档收录2026KPL职业联赛夏季赛英雄热度、登场率、胜率和Ban率页面截图。",
           [2], "volatile_snapshot", captured_at="2026-09-07",
           note="截图标明“算法测试中”，样本场次较小且OCR不能保证逐项准确；不录入英雄强度或固定分路。"),
    _entry("equipment_shop_snapshot", "版本资料", "装备商店截图",
           "文档收录攻击、法术、防御、移动、打野和辅助装备的商店快照。",
           [3], "volatile_snapshot",
           note="装备名称、价格、属性和合成路径随补丁变化；本轮只登记资料存在，不据此校验录像。"),
    _entry("summoner_spell_snapshot", "版本资料", "召唤师技能截图",
           "文档截图列出治疗、晕眩、惩击、干扰、净化、终结、疾跑、狂暴、闪现、传送和弱化。",
           [4], "volatile_snapshot",
           note="技能列表和效果必须按录像实际补丁复核。"),
    _entry("smite_snapshot", "版本资料", "惩击效果快照",
           "截图显示惩击冷却30秒，对附近野怪和小兵造成1500真实伤害并眩晕1秒。",
           [4], "volatile_snapshot",
           note="非官方截图且数值可能随打野刀、技能升级或补丁改变，不接入伤害校验。"),
]


def payload() -> dict[str, Any]:
    entries = deepcopy(ENTRIES)
    return {
        **META,
        "entries": entries,
        "entryCount": len(entries),
        "termCount": sum(1 for item in entries if item["annotationAliasAllowed"]),
        "overgeneralizedCount": sum(1 for item in entries if item["status"] == "overgeneralized"),
        "volatileCount": sum(1 for item in entries if item["status"] == "volatile_snapshot"),
        "modelActiveCount": sum(1 for item in entries if item["modelFeatureAllowed"]),
    }


def normalize_term(text: Any) -> dict[str, Any] | None:
    """把人工标注中的通俗说法归一化，同时保留非官方来源状态。"""
    if not isinstance(text, str):
        return None
    candidate = "".join(text.strip().lower().split())
    if not candidate:
        return None
    for entry in ENTRIES:
        if not entry["annotationAliasAllowed"]:
            continue
        for alias in entry["aliases"]:
            if candidate == "".join(alias.lower().split()):
                return {
                    "termId": entry["id"],
                    "label": entry["title"],
                    "input": text,
                    "sourceStatus": "non_official_personal_summary",
                    "confidence": META["defaultConfidence"],
                    "validationAllowed": False,
                    "modelFeatureAllowed": False,
                }
    return None
