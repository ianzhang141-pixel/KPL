"""S44 赛季规则知识库。

来源是用户在 2026-09-08 提供的当前赛季规则说明。`user_confirmed` 表示
“用户已核对”，不等于已用官方公告二次核验；不完整或疑似笔误的条目不会
悄悄变成可计算常量。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


META = {
    "season": "S44",
    "label": "S44（当前赛季）",
    "providedAt": "2026-09-08",
    "provenance": "user-provided",
    "externalVerified": False,
}


def _rule(
    rule_id: str,
    category: str,
    title: str,
    summary: str,
    values: dict[str, Any],
    status: str = "user_confirmed",
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "category": category,
        "title": title,
        "summary": summary,
        "values": values,
        "status": status,
        "machineActive": status == "user_confirmed",
        "note": note,
    }


RULES = [
    _rule("jungle_protection", "0分钟", "野区保护",
          "0:00–4:00，敌方英雄在我方野区造成的伤害降低15%。",
          {"startSec": 0, "endSec": 240, "damageReductionPct": 15}),
    _rule("tower_early_protection", "0分钟", "一塔前期保护",
          "0:00–4:00，一塔受到的伤害额外降低40%。另有一次护盾触发描述待澄清。",
          {"startSec": 0, "endSec": 240, "damageReductionPct": 40}),
    _rule("tower_lost_health_shield", "0分钟", "防御塔已损生命护盾",
          "原文涉及损失25%最大生命值后触发、护盾等于已损生命值100%，语句存在歧义。",
          {"beforeSec": 240, "triggerLostMaxHpPct": 25,
           "shieldOfLostHpPct": 100}, "ambiguous",
          "未确认触发次数、持续时间和原文中的疑似错字，因此不参与计算。"),

    _rule("crossbow_minion", "4分钟", "弩车加入兵线",
          "4:00起，每波一个远程法师小兵替换为弩车。",
          {"fromSec": 240, "replacesPerWave": 1, "replaces": "ranged_caster"}),
    _rule("overlord", "4分钟", "主宰",
          "4:00首次出现，重生间隔4分钟；击杀后本方三路接下来2波替换为主宰先锋。",
          {"firstSpawnSec": 240, "respawnSec": 240, "affectedLanes": 3,
           "replacedWaves": 2, "vanguardTowerDamageReductionPct": 50}),
    _rule("tyrant", "4分钟", "暴君",
          "4:00首次出现，重生间隔3分30秒；击杀后全队获得君临闪电链。",
          {"firstSpawnSec": 240, "respawnSec": 210, "cooldownSec": 3,
           "heroDamageBase": 20, "heroDamagePerLevel": 10,
           "nonHeroDamageBase": 60, "nonHeroDamagePerLevel": 30,
           "bounceRadius": 400, "maxTargets": 3, "damagesTowers": False,
           "triggeredBy": ["basic_attack", "skill"]}),

    _rule("cannon_minion", "10分钟", "炮车加入兵线",
          "10:00起炮车替换弩车，攻击和生命高于弩车，并优先持续攻击射程内防御塔。",
          {"fromSec": 600, "replaces": "crossbow", "towerDamagePct": 150,
           "towerDamageTakenPct": 70}),
    _rule("shadow_tyrant", "10分钟", "暗影暴君",
          "10:00出现，重生间隔3分30秒；提供经验、金币和加强版君临。",
          {"firstSpawnSec": 600, "respawnSec": 210,
           "buff": "stronger_junlin", "exactBuffDamage": None},
          "incomplete", "加强版君临的具体伤害数值未提供，不参与伤害计算。"),
    _rule("shadow_overlord", "10分钟", "暗影主宰",
          "10:00出现，重生间隔3分30秒；强化2波三路兵线，击杀者获得一次召唤暗影先锋。",
          {"firstSpawnSec": 600, "respawnSec": 210, "affectedLanes": 3,
           "replacedWaves": 2, "summonSkillUses": 1,
           "summonClearsSelectedLane": True}),

    _rule("storm_dragon", "20分钟", "风暴龙王",
          "20:00降临，重生3分钟；雷击、脱战刷新护盾，并强化接下来2波兵线。",
          {"firstSpawnSec": 1200, "respawnSec": 180,
           "heroTrueDamageMaxHpPct": 5, "nonHeroTrueDamageMaxHpPct": 20,
           "shieldStartMaxHpPct": 20, "shieldGrowthEverySec": 30,
           "shieldGrowthPct": 1.5, "shieldMaxMaxHpPct": 50,
           "replacedWaves": 2, "towerDisableSec": 5,
           "affectsAllFriendlyUnits": True}),

    _rule("primal_bond", "中立资源", "原初羁绊",
          "击败暴君或主宰后获得90秒减益，对暴君、主宰伤害降低50%。",
          {"durationSec": 90, "objectiveDamageReductionPct": 50,
           "triggersHeavyObjectiveSkill": True}),
    _rule("minion_waves", "兵线", "兵线刷新与初始组成",
          "兵线每33秒刷新；三路初始均为1近战+2远程法师兵。首批10秒登场。",
          {"firstSpawnSec": 10, "respawnSec": 33, "meleePerWave": 1,
           "rangedPerWave": 2,
           "clashLaneRangedType": "experience", "midLaneRangedType": "normal",
           "farmLaneRangedType": "gold"}),
    _rule("lane_early_bonus", "兵线", "分路小兵前期额外收益",
          "4分钟前，发育路法师兵额外40金币；对抗路法师兵额外65经验。",
          {"endSec": 240, "farmLaneExtraGold": 40,
           "clashLaneExtraExperience": 65}),
    _rule("minion_speed", "兵线", "10分钟后兵线加速",
          "10:00起加速；中路350→450（18分钟达到）。边路原文写385→495于180分钟达到，疑似18分钟。",
          {"fromSec": 600, "midStartSpeed": 350, "midEndSpeed": 450,
           "midEndSec": 1080, "sideStartSpeed": 385, "sideEndSpeed": 495,
           "sideEndSecCandidate": 1080, "sourceSideEndText": "180分钟"},
          "ambiguous", "边路结束时间暂按疑似笔误保留候选值，但不参与计算。"),
    _rule("minion_late_scaling", "兵线", "20分钟兵线强化",
          "20分钟后兵线属性大幅加强，推进能力提升。",
          {"fromSec": 1200, "exactStats": None}, "incomplete",
          "未提供具体生命值和攻击成长。"),
    _rule("super_minion", "兵线", "超级兵",
          "某路高地塔被摧毁后，下一波起该路每波弩车或炮车替换为超级兵。",
          {"startsNextWave": True, "laneSpecific": True,
           "replaces": ["crossbow", "cannon"]}),

    _rule("tower_destroy_slow", "防御塔", "防御塔被击毁时减速兵线",
          "塔被击毁后减速范围内敌方小兵3秒：一塔30%、二塔60%、高地塔90%。",
          {"durationSec": 3, "outerTowerSlowPct": 30,
           "secondTowerSlowPct": 60, "highGroundTowerSlowPct": 90}),
    _rule("tower_dive_protection", "防御塔", "越塔保护",
          "0:00–4:00塔下敌方英雄输出降低25%，4:00后10%，10:00后5%。",
          {"stages": [{"fromSec": 0, "toSec": 240, "damageReductionPct": 25},
                      {"fromSec": 240, "toSec": 600, "damageReductionPct": 10},
                      {"fromSec": 600, "toSec": None, "damageReductionPct": 5}]}),
    _rule("tower_active_defense", "防御塔", "主动防御",
          "连续攻击同一英雄时，下次伤害增加60%，最多增加300%，并无视防御。",
          {"damageGrowthPerHitPct": 60, "maxDamageGrowthPct": 300,
           "ignoresDefense": True}),
    _rule("tower_backdoor", "防御塔", "反偷塔与通用特性",
          "攻击范围内无敌方兵线时受到伤害额外降低55%；提供真实视野；英雄伤害不能暴击防御塔。",
          {"noEnemyWaveDamageReductionPct": 55, "grantsTrueSight": True,
           "heroDamageCanCrit": False}),

    _rule("primordial_circle", "地图机制", "原初法阵与空间之灵",
          "1:00空间之灵出现，60秒刷新，4:00后停止；传送阵10:00消失。",
          {"spiritFirstSpawnSec": 60, "spiritRespawnSec": 60,
           "spiritStopsAfterSec": 240, "portalDisappearsSec": 600,
           "teamSharedGold": 50, "channelSec": 3, "reuseCooldownSec": 30,
           "maxSameTeamUsers": 2, "targets": ["friendly_tower", "friendly_minion"]}),
    _rule("vision_spirit", "地图机制", "二塔视野之灵",
          "边路二塔被摧毁后在遗址生成，持续60秒、重生20秒；驱赶敌方视野之灵获得1金币。",
          {"trigger": "side_second_tower_destroyed", "durationSec": 60,
           "respawnSec": 20, "enemyDispelGold": 1,
           "revealsLargeJungleArea": True, "enemyMustEnterBushToSee": True}),

    _rule("red_blue_buff_common", "野怪", "红蓝石像通用规则",
          "0:30首次出现，90秒重生；击杀回复225生命，增益持续70秒，被击败后转移并刷新。",
          {"firstSpawnSec": 30, "respawnSec": 90, "heal": 225,
           "buffDurationSec": 70, "transfersOnHolderKilled": True}),
    _rule("blue_buff", "野怪", "蔚蓝石像之力",
          "技能冷却缩短20%，每秒额外回复3%法力值。",
          {"cooldownReductionPct": 20, "manaRegenPctPerSec": 3}),
    _rule("red_buff", "野怪", "猩红石像之力",
          "普攻15%减速、最多2层、持续2秒；灼烧持续2秒。",
          {"slowPctPerStack": 15, "maxSlowStacks": 2, "slowDurationSec": 2,
           "burnDurationSec": 2, "monsterTrueDamageRange": [60, 256],
           "nonMonsterTrueDamageRange": [30, 128]}),
    _rule("red_falcon", "野怪", "红隼",
          "0:30首次出现，位于发育路裂口；死亡后不再刷新，受击会周期性产生击退气流。",
          {"firstSpawnSec": 30, "respawns": False, "teamGoldSourceText": "25～20"},
          "ambiguous", "团队每人金币原文为25～20，区间方向或随时间变化规则未说明。"),
]


STATUS_LABELS = {
    "user_confirmed": "用户已核对",
    "incomplete": "数值不完整",
    "ambiguous": "存在歧义",
}


def payload() -> dict[str, Any]:
    rules = deepcopy(RULES)
    for entry in rules:
        entry["statusLabel"] = STATUS_LABELS[entry["status"]]
    return {
        **META,
        "rules": rules,
        "ruleCount": len(rules),
        "machineActiveCount": sum(1 for item in rules if item["machineActive"]),
        "needsReviewCount": sum(1 for item in rules if not item["machineActive"]),
    }
