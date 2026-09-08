"""S44 经济机制参考库。

来源为用户提供的《S44王者经济机制》文档。文档明确声明是业余实测、非官方
数据，因此这里的规则不能覆盖用户提供的 S44 官方说明，也不能作为硬性
合法性校验。内部一致且公式完整的条目可以用于带来源标记的估算；冲突、模糊
或只在训练营/指挥官模式出现的条目必须隔离。
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


META = {
    "season": "S44",
    "label": "S44 经济机制参考库",
    "document": "S44王者经济机制.docx",
    "providedAt": "2026-09-09",
    "provenance": "user-provided-non-official-testing-notes",
    "externalVerified": False,
    "official": False,
    "defaultConfidence": 0.55,
    "authorityRank": 20,
    "usagePolicy": "仅用于带来源标记的估算；不得覆盖S44官方说明或充当训练真值。",
}


STATUS_LABELS = {
    "reference_calculable": "参考公式可计算",
    "reference_only": "仅作参考",
    "conflict": "与现有规则冲突",
    "superseded_by_official": "已按官方说明停用",
    "ambiguous": "条件不完整",
    "out_of_scope": "非标准对局范围",
}


def _entry(
    entry_id: str,
    category: str,
    title: str,
    summary: str,
    values: dict[str, Any],
    pages: list[int],
    status: str = "reference_only",
    note: str = "",
) -> dict[str, Any]:
    if status not in STATUS_LABELS:
        raise ValueError(f"未知经济规则状态：{status}")
    return {
        "id": entry_id,
        "category": category,
        "title": title,
        "summary": summary,
        "values": values,
        "sourcePages": pages,
        "status": status,
        "statusLabel": STATUS_LABELS[status],
        "estimateAllowed": status == "reference_calculable",
        "validationAllowed": False,
        "confidence": META["defaultConfidence"] if status == "reference_calculable" else 0.0,
        "note": note,
    }


ENTRIES = [
    _entry("taiyi_passive", "英雄与装备", "太乙真人被动经济",
           "文档称太乙真人及800范围内友军击杀非英雄单位时，相关英雄额外获得10%金币。",
           {"radius": 800, "extraGoldPct": 10, "target": "non_hero_units"}, [1],
           "reference_only", "英雄技能会随版本调整，必须从录像确认英雄与作用范围后才能使用。"),
    _entry("shaosiyuan_economy", "英雄与装备", "少司缘技能经济",
           "文档称部分良缘/冤缘触发给予20金币，二技能还包含暂扣及返还金币机制。",
           {"performanceGold": 20, "activeHighHealthHalfReward": True,
            "temporaryPenaltyGoldPerTick": 5, "refundOnKillPct": 100,
            "refundOnAssistPct": 50}, [1], "reference_only",
           "触发次数、结算边界及技能版本需要用真实录像复核，暂不自动累计。"),
    _entry("dolia_ultimate_income_share", "英雄与装备", "朵莉亚大招收益平分",
           "文档技能截图称，被刷新技能在5秒内使用时，该技能造成的击败收益与朵莉亚平分。",
           {"useWindowSec": 5, "shareParticipants": 2}, [6], "reference_only",
           "只记录条件，不在未识别技能来源时推算经济。"),
    _entry("jungle_support_item_rules", "英雄与装备", "打野刀与辅助装经济规则",
           "文档称辅助装不参与兵线/野怪经济分配；二、三级打野刀的野怪金币倍率为120%。",
           {"supportSharesLaneGold": False, "supportSharesJungleGold": False,
            "tier2JungleGoldMultiplier": 1.2, "tier3JungleGoldMultiplier": 1.2}, [1],
           "reference_only", "前4分钟二级打野刀还有辅助在场例外，已单独记录。"),
    _entry("opening_gold", "基础经济", "开局经济",
           "每名英雄开局获得300金币。", {"gold": 300}, [1],
           "reference_calculable"),
    _entry("passive_gold", "基础经济", "自然增长经济",
           "从游戏时间0:30起（包含第30秒），每秒获得3金币。",
           {"startsAtSec": 30, "goldPerSec": 3, "inclusiveFirstTick": True}, [1],
           "reference_calculable", "文档示例3:30累计543金币，公式内部一致。"),
    _entry("unit_income_split", "兵线与野怪", "多人分配与补刀",
           "1至5人共享基础经济的个人比例依次为100%、80%、55%、43%、37%；补刀者另得50%基础经济。",
           {"sharePctByParticipants": {1: 100, 2: 80, 3: 55, 4: 43, 5: 37},
            "lastHitBonusPct": 50, "roundingToleranceGold": 1}, [1],
           "reference_calculable", "文档称实测金币可能上下波动约1金币。"),
    _entry("early_jungle_knife_lane_exception", "兵线与野怪", "前4分钟二级打野刀分线例外",
           "文档称二级打野刀前4分钟通常不分兵线经济；辅助装英雄在旁时该限制失效。",
           {"beforeSec": 240, "tier2JungleKnifeNormallySharesLane": False,
            "supportNearbyException": True}, [1], "reference_only",
           "依赖装备版本与站位，尚未用真实S44录像复核。"),
    _entry("income_radius", "兵线与野怪", "基础经济获取距离",
           "文档将无二级打野刀的兵线/野怪基础经济极限距离记为约1000码，并称700至1000码存在打野刀例外。",
           {"baseIncomeRadiusApprox": 1000, "jungleKnifeFullEffectRadiusApprox": 700}, [1],
           "reference_only", "距离来自训练营描述，不作为坐标模型硬阈值。"),
    _entry("wave_clock", "兵线", "兵线刷新与交汇公式",
           "首波0:10刷新，之后每33秒一波；10分钟前首波约0:30交汇，之后每33秒交汇。",
           {"spawnFormulaSec": "33*n-23", "clashFormulaBefore10MinSec": "33*n-3",
            "firstSpawnSec": 10, "respawnSec": 33}, [1],
           "reference_calculable", "刷新时间与S44官方说明一致；交汇时间仍是非官方参考值。"),
    _entry("early_mid_wave_conflict", "兵线", "前4分钟中路兵线组成",
           "文档称中路为2近战+2远程；S44官方说明为1近战+2普通法师兵。",
           {"document": {"melee": 2, "ranged": 2},
            "official": {"melee": 1, "ranged": 2}}, [2], "superseded_by_official",
           "采用官方说明；非官方文档数值已停用，不得参与计算。"),
    _entry("early_lane_wave_gold", "兵线", "前7波三路单人补刀经济",
           "文档记录前7波对抗路、中路、发育路单人补刀总经济序列。",
           {"clashLane": [190, 190, 194, 199, 204, 210, 211],
            "midLane": [208, 208, 212, 218, 230, 236, 238],
            "farmLane": [210, 210, 212, 217, 226, 230, 237]}, [2],
           "reference_only", "原始文档称训练营实测且单兵可能波动2金币以内。"),
    _entry("mid_game_wave_gold", "兵线", "4至10分钟兵线经济",
           "文档称第8至18波三路经济相同，单人补刀总经济从238逐步增长至286。",
           {"waveNumberFrom": 8, "waveNumberTo": 18,
            "singleLastHitTotals": [238, 242, 245, 250, 254, 260, 264, 272, 278, 281, 286]}, [2],
           "reference_only"),
    _entry("late_wave_gold", "兵线", "10分钟后普通兵线经济",
           "文档称第19波起非超级兵单人补刀总经济约290，之后不再成长。",
           {"fromWave": 19, "fromSec": 604, "singleLastHitTotal": 290,
            "components": [85, 73, 132]}, [2], "reference_only",
           "与用户提供的20分钟兵线属性强化不冲突；文档只声称金币不再成长。"),
    _entry("neutral_spawn_reference", "野怪", "非龙野怪刷新参考",
           "红蓝BUFF和红隼0:30出现，空间之灵1:00出现，其他小野0:30出现并在击杀后70秒刷新。",
           {"buffFirstSpawnSec": 30, "buffRespawnSec": 90,
            "redFalconFirstSpawnSec": 30, "redFalconRespawns": False,
            "spiritFirstSpawnSec": 60, "spiritRespawnSec": 60,
            "otherJungleFirstSpawnSec": 30, "otherJungleRespawnSec": 70}, [3],
           "reference_only", "红蓝BUFF、红隼和空间之灵时间与现有规则一致；其他小野仍待外部复核。"),
    _entry("jungle_gold", "野怪", "普通野怪单人补刀经济",
           "文档记录携带二/三级打野刀时，普通野怪在16分钟前与成长完成后的经济。",
           {"beforeGrowth": {"buff": 119, "lizardOrMonkey": 78,
                              "wolves": [39, 39], "boars": [54, 24], "redArmorOrBird": 58},
            "afterGrowth": {"buff": 130, "lizardOrMonkey": 83,
                             "wolves": [40, 40], "boars": [57, 26], "redArmorOrBird": 62},
            "growthStartsSec": 960, "growthEndsSec": 1050,
            "includesJungleKnifeMultiplier": True}, [3], "reference_only",
           "训练营数据且包含打野刀倍率，不能直接当作野怪基础金币。"),
    _entry("red_falcon_gold_conflict", "野怪", "红隼团队金币",
           "文档称队友固定各得20金币，击杀者另有随时间成长的补刀经济；现有用户原文为每人25至20。",
           {"documentTeamGoldPerAlly": 20, "documentLastHitGoldRange": [64, 90],
            "officialSourceText": "25～20"}, [3], "superseded_by_official",
           "采用官方原文“25～20”；因变化方向或条件不明，仍保持不可计算。"),
    _entry("spirit_gold", "野怪", "空间之灵团队经济",
           "文档称击杀者基础经济随时间增长，且全队每人获得10金币；与现有全队合计50金币一致。",
           {"teamGoldPerHero": 10, "teamGoldTotal": 50,
            "lastHitGoldByMinute": {1: 55, 2: 60, 3: 60, 4: 61, 5: 61,
                                    6: 64, 7: 66, 8: 69, 9: 70, 10: 72, 11: 75}}, [3],
           "reference_only", "标准对局只在4分钟前刷新；后续分钟数据可能来自延迟击杀或训练模式。"),
    _entry("tyrant_respawn_conflict", "龙", "普通暴君重生间隔",
           "文档把暴君和主宰都记为击杀后4分钟刷新；S44官方说明为暴君3分30秒、主宰4分钟。",
           {"documentTyrantRespawnSec": 240, "officialTyrantRespawnSec": 210,
            "documentAndOfficialOverlordRespawnSec": 240}, [3], "superseded_by_official",
           "采用官方说明：暴君210秒、主宰240秒。"),
    _entry("objective_economy", "龙", "龙的补刀与全队经济参考",
           "文档记录携带二/三级打野刀的击杀者补刀经济和4名队友低保经济。",
           {"overlord": {"lastHit": 40, "ally": 20, "teamTotal": 120},
            "shadowTyrant": {"lastHit": 163, "ally": 83, "teamTotal": 495},
            "shadowOverlord": {"lastHit": 61, "ally": 31, "teamTotal": 185},
            "stormDragon": {"lastHit": 204, "ally": 100, "teamTotal": 604},
            "includesJungleKnifeMultiplierForLastHit": True}, [3, 4], "reference_only",
           "低保金币存在约2金币波动，且打野刀是否影响各角色需保留条件。"),
    _entry("tyrant_gold_growth", "龙", "普通暴君经济成长参考",
           "文档称普通暴君经济随击杀时间增长，4:00总计约157，9:00后约342。",
           {"samples": {240: [53, 26], 270: [61, 31], 300: [66, 33],
                        330: [73, 37], 360: [78, 39], 390: [84, 42],
                        420: [91, 46], 450: [96, 48], 480: [103, 52],
                        510: [108, 54], 540: [114, 57], 565: [114, 57]},
            "sampleFormat": ["lastHitWithJungleKnife", "eachAlly"]}, [4],
           "reference_only"),
    _entry("tower_gold", "防御塔", "防御塔全队与范围内补刀经济",
           "全队每人获得基础金币；塔攻击范围内英雄再平分相当于50%基础金币的额外奖励。",
           {"baseGold": {"outer": 50, "second": 65, "highGround": 80},
            "nearbyBonusPoolPctOfBase": 50, "crystalGold": 0}, [4],
           "reference_calculable", "补刀奖励按范围内人数平分，与最后一击和辅助装无关。"),
    _entry("vision_spirit_respawn_conflict", "防御塔", "二塔视野之灵重生时间",
           "文档记为持续60秒、120秒刷新；S44官方说明记为20秒重生。",
           {"durationSec": 60, "documentRespawnSec": 120,
            "officialRespawnSec": 20, "gold": 1}, [4], "superseded_by_official",
           "采用官方说明的20秒；1金币奖励两边资料一致。"),
    _entry("catchup_thresholds", "人头", "8%经济追赶机制",
           "文档称达到分时段团队经济差阈值后，优势方人头收益乘0.92，劣势方乘1.08。",
           {"advantagedMultiplier": 0.92, "disadvantagedMultiplier": 1.08,
            "thresholds": [[0, 240, 1500], [240, 480, 2000], [480, 600, 2500],
                           [600, 720, 3000], [720, 840, 3500], [840, 960, 4000],
                           [960, 1020, 4500], [1020, 1080, 5000], [1080, 1140, 5500],
                           [1140, 1200, 6000], [1200, None, 7000]]}, [5],
           "reference_only", "阈值是训练营反推值，不作为官方规则。"),
    _entry("kill_base_gold", "人头", "英雄等级对应基础人头经济",
           "无悬赏、无连死衰减时，1至3级150，4至7级175，8至15级200。",
           {"levelBands": [[1, 3, 150], [4, 7, 175], [8, 15, 200]],
            "firstBloodBonus": 75}, [5, 6], "reference_calculable"),
    _entry("bounty_gold", "人头", "悬赏经济档位",
           "文档按等级段记录本体经济与全队低保悬赏金币；同一二至四级悬赏各有两个内部档位。",
           {"lowGuaranteeTiers": [0, 20, 35, 50, 75, 100, 125, 150],
            "bodyGoldByLevelBand": {
                "1-3": [150, 150, 165, 180, 210, 240, 270, 300],
                "4-7": [175, 175, 192, 210, 245, 280, 315, 350],
                "8-15": [200, 200, 220, 240, 280, 320, 360, 400],
            }}, [5], "reference_only",
           "悬赏内部档位的触发积分未给出，不能从画面外推档位。"),
    _entry("death_decay", "人头", "连续阵亡经济衰减",
           "文档记录连续阵亡时本体经济逐次下降，击杀英雄后恢复。",
           {"1-3": [150, 135, 112, 90, 67, 45, 30, 15],
            "4-7": [175, 157, 131, 105, 78, 52, 35, 17],
            "8-15": [200, 180, 150, 120, 90, 60, 40, 20]}, [5, 6],
           "reference_only", "劣势方最低经济还存在条件未知的突变，故不开放完整计算。"),
    _entry("kill_assist_formulas", "人头", "击杀、助攻与激励金币包",
           "击杀收益包含本体、一血和悬赏低保；助攻按本体的50%除以助攻人数，并可能加入30%本体的激励金币包。",
           {"assistBodyPoolPct": 50, "assistIncentivePct": 30,
            "catchupMultiplierAppliedLast": True}, [6], "reference_only",
           "原初试炼积分与贡献最大者判定条件未公开，暂不自动分配激励金币包。"),
    _entry("level_25_reference", "人头", "15至25级描述",
           "文档写到15至25级沿用同一人头经济表，但标准S44对局当前按最高15级处理。",
           {"documentMaxReferencedLevel": 25, "standardModelMaxLevel": 15}, [5, 6],
           "out_of_scope", "推定来自训练营或指挥官模式，不进入标准比赛模型。"),
    _entry("highlight_gold", "其他经济", "高光经济",
           "文档称关键高光60金币、完美高光120金币；普通多杀、团灭等播报本身无金币。",
           {"keyHighlightGold": 60, "perfectHighlightGold": 120,
            "ordinaryAnnouncementGold": 0}, [7], "reference_only",
           "高光判定依赖平台内部统计，录像画面未必能完整复原。"),
    _entry("support_item_gold", "其他经济", "辅助装低保与拾取",
           "一、二级辅助装每3秒5金币，三级每3秒6金币；二级最多拾取450，三级最多600。",
           {"passiveGoldPer3Sec": {"tier1": 5, "tier2": 5, "tier3": 6},
            "pickupCap": {"tier2": 450, "tier3": 600},
            "pickupIntervalSec": {"tier2": 15, "tier3": 10}}, [7],
           "reference_only", "文档也指出训练营与真实排位的低保目标规则存在差异。"),
    _entry("summoned_unit_gold", "其他经济", "傀儡与召唤物经济",
           "文档记录元歌傀儡60、空空儿傀儡6，以及部分召唤物3至12金币。",
           {"yuanGePuppet": 60, "kongKongErPuppet": 6,
            "miladyRobots": [3, 6, 12], "cangWolf": 3, "mengTianGuard": 3}, [7],
           "reference_only", "英雄与召唤物版本变化频繁，只作录像解释参考。"),
]


def payload() -> dict[str, Any]:
    entries = deepcopy(ENTRIES)
    return {
        **META,
        "entries": entries,
        "entryCount": len(entries),
        "calculableCount": sum(1 for item in entries if item["estimateAllowed"]),
        "conflictCount": sum(1 for item in entries if item["status"] == "conflict"),
        "supersededCount": sum(1 for item in entries
                               if item["status"] == "superseded_by_official"),
        "excludedCount": sum(1 for item in entries
                             if item["status"] in {"conflict", "superseded_by_official",
                                                   "ambiguous", "out_of_scope"}),
    }


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}必须是数字。") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name}必须是有限的非负数。")
    return number


def natural_gold_at(at_sec: Any) -> dict[str, Any]:
    """按文档公式估算截至某秒的自然金币，不包含开局300金币。"""
    at = _finite_nonnegative(at_sec, "游戏时间")
    ticks = max(0, math.floor(at) - 29)
    return {
        "gold": ticks * 3,
        "ticks": ticks,
        "sourceRule": "passive_gold",
        "sourceStatus": "non_official_reference",
        "confidence": META["defaultConfidence"],
    }


def shared_unit_gold(base_gold: Any, participants: int, last_hitter: int = 1) -> dict[str, Any]:
    """估算多人吃线/吃野的理论分配；返回原始值并明确1金币波动范围。"""
    base = _finite_nonnegative(base_gold, "基础经济")
    if participants not in {1, 2, 3, 4, 5}:
        raise ValueError("参与人数必须是1至5。")
    if not 1 <= last_hitter <= participants:
        raise ValueError("补刀者编号必须属于参与者。")
    share_pct = {1: 100, 2: 80, 3: 55, 4: 43, 5: 37}[participants]
    normal = base * share_pct / 100
    last_hit = normal + base * 0.5
    values = [last_hit if index == last_hitter else normal
              for index in range(1, participants + 1)]
    return {
        "perHeroRaw": values,
        "teamTotalRaw": sum(values),
        "roundingToleranceGold": 1,
        "sourceRule": "unit_income_split",
        "sourceStatus": "non_official_reference",
        "confidence": META["defaultConfidence"],
    }


def tower_gold(tower: str, nearby_heroes: int, team_size: int = 5) -> dict[str, Any]:
    """估算拆塔团队金币；基础金币全队获得，额外池只由塔范围内英雄平分。"""
    bases = {"outer": 50, "second": 65, "highGround": 80}
    if tower not in bases:
        raise ValueError("防御塔必须是outer、second或highGround。")
    if not 1 <= nearby_heroes <= team_size <= 5:
        raise ValueError("范围内人数与队伍人数不合法。")
    base = bases[tower]
    extra_pool = base * 0.5
    nearby = base + extra_pool / nearby_heroes
    return {
        "nearbyHeroRaw": nearby,
        "remoteHeroRaw": base,
        "teamTotalRaw": nearby * nearby_heroes + base * (team_size - nearby_heroes),
        "sourceRule": "tower_gold",
        "sourceStatus": "non_official_reference",
        "confidence": META["defaultConfidence"],
    }
