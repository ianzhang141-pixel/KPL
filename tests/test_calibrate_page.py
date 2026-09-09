"""标定页的几条契约。

页面是纯 HTML + 原生 JS，没有构建步骤，所以这里测的是**文本层面的契约**：
哪些东西必须出现在页面里、哪些绝不能再出现。看着笨，但它挡住的是
三个已经真实发生过、而且后端完全查不出来的故障：

1. **地标选不中。** 以前 26 个地标藏在一个 ``<select id="lmSel">`` 里，
   只靠 ``change`` 事件调用 pickLm。用户想标的如果正好是下拉框里当前那一项，
   点它**不会**触发 change —— pickLm 不跑，LM_ACTIVE 还是空的，
   于是拖出来的框落到了上一次选中的普通项目上。
   表现就是「防御塔／资源坑／头像根本标不了」，而后端一切正常。
   所以：**每个地标都必须有自己的点击入口，不能只有一个下拉框。**

2. **隐藏被写成了删除。** 「框太多看不清地图」的正确解法是不画出来，
   不是把坐标扔掉。隐藏逻辑里出现 ``delete`` 就是把两件事混了。

3. **旋钮没人看得懂。** 「蓝 B-R」这种标签只有写代码的人看得懂。
   小地图阈值页面上每一个能填的数字，都必须配一句人话解释。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from kplab import events_cv, minimap

PAGE = Path(__file__).resolve().parents[1] / "kplab" / "web" / "calibrate.html"
HTML = PAGE.read_text(encoding="utf-8")

# 页面上能填的小地图阈值。default_thresholds() 里还有 verified/note 两个非数值项。
# bool 是 int 的子类，verified 会被 isinstance(v, int) 放进来，必须单独挡掉。
TUNABLE = [k for k, v in minimap.default_thresholds().items()
           if isinstance(v, int) and not isinstance(v, bool)]


class TestLandmarksAreClickable(unittest.TestCase):
    """契约一：26 个地标每一个都要能直接点中。"""

    def test_no_select_only_landmark_picker(self):
        self.assertNotIn(
            'id="lmSel"', HTML,
            "地标不能再退回下拉框：选中项被再次点击时 change 不触发，"
            "pickLm 不会跑，用户拖的框会落到别的项目上。",
        )

    def test_every_landmark_renders_its_own_pick_button(self):
        # 生成 chip 的那一行必须同时带上 pick 按钮和 pickLm 调用。
        self.assertRegex(
            HTML, r'class="pick"\s+onclick="pickLm\(',
            "每个地标方块都要自带一个调用 pickLm 的按钮。",
        )

    def test_pick_clears_the_other_kind_of_target(self):
        # 同一次拖框只能落到一个目标上。两个 pick 函数必须互相清空。
        self.assertRegex(HTML, r"function pickLm\([^)]*\)\{\s*\n?\s*LM_ACTIVE=n; ACTIVE=null;")
        self.assertRegex(HTML, r"function pickRegion\([^)]*\)\{\s*\n?\s*ACTIVE=k; LM_ACTIVE=null;")

    def test_all_three_landmark_kinds_are_grouped_on_the_page(self):
        for kind in (events_cv.KIND_TOWER, events_cv.KIND_OBJECTIVE, events_cv.KIND_PORTRAIT):
            self.assertIn(f"['{kind}',", HTML,
                          f"地标分组少了 {kind} —— 这一类在页面上就没有入口了。")

    def test_pits_match_the_backend(self):
        # 前端写死的坑名必须和 events_cv 认的一致，否则标了也检测不到。
        for name in events_cv.OBJECTIVE_PITS:
            self.assertIn(f"name:'{name}'", HTML,
                          f"页面上没有 {name}，后端却在找它。")


class TestHidingIsNotDeleting(unittest.TestCase):
    """契约二：隐藏只影响画不画，绝不动坐标。"""

    def _body(self, name: str) -> str:
        match = re.search(rf"function {name}\(.*?\n\}}", HTML, re.S)
        self.assertIsNotNone(match, f"找不到 {name}()")
        return match.group(0)

    def test_hide_helpers_never_delete_data(self):
        for name in ("toggleHidden", "hideAll", "showAll", "onlyCurrent",
                     "hideAllLm", "showAllLm", "setGroupHidden"):
            body = self._body(name)
            self.assertNotIn(
                "delete ", body,
                f"{name}() 里出现了 delete —— 隐藏被写成了删除，标定结果会丢。",
            )
            self.assertNotIn("LANDMARKS[", body)
            self.assertNotIn("REGIONS[", body)

    def test_overlay_skips_hidden_but_loops_over_all_data(self):
        body = self._body("drawOverlay")
        self.assertIn("isHidden(rKey(k))", body)
        self.assertIn("isHidden(lKey(n))", body)
        # 仍然遍历完整的数据，只是跳过不画 —— 数据没有被过滤掉。
        self.assertIn("Object.entries(LANDMARKS)", body)

    def test_picking_a_hidden_target_unhides_it(self):
        # 否则用户选中一个被隐藏的地标，拖了框却什么都看不见，会以为坏了。
        self.assertIn("setHidden(lKey(n), false)", HTML)
        self.assertIn("setHidden(rKey(k), false)", HTML)


class TestThresholdsAreExplained(unittest.TestCase):
    """契约三：每个能填的数字都要有人话解释。"""

    def test_every_tunable_has_an_input(self):
        for key in TUNABLE:
            self.assertIn(f'id="{key}"', HTML, f"页面上没有 {key} 的输入框。")

    def test_every_tunable_input_is_followed_by_a_help_line(self):
        for key in TUNABLE:
            match = re.search(rf'id="{key}">(.*?)</div>', HTML, re.S)
            self.assertIsNotNone(match, f"{key} 的输入框结构变了")
            self.assertIn('class="help"', match.group(1),
                          f"{key} 后面没有解释这个数字是什么意思的说明。")

    def test_no_cryptic_channel_labels_remain(self):
        for cryptic in ("蓝 B-R", "蓝 B-G", "红 R-B", "红 R-G", "蓝 最低B", "红 最低R"):
            self.assertNotIn(cryptic, HTML,
                             f"「{cryptic}」这种标签只有写代码的人看得懂。")

    def test_the_explainer_states_the_actual_rule(self):
        # 说明里写的判定条件必须和 classify_pixels 真正做的事一致：
        # 三条同时满足（≥ 最低亮度、比另一主色多、比绿色多）。
        self.assertIn("同时", HTML)
        self.assertIn("蓝色分量 − 红色分量", HTML)
        self.assertIn("蓝色分量 − 绿色分量", HTML)

    def test_reset_button_uses_backend_defaults(self):
        # 「恢复默认」不能在前端另写一份数字，否则两边会漂。
        self.assertIn("MM_DEFAULTS", HTML)
        self.assertIn("st.minimapDefaults", HTML)


class TestStatusExposesDefaults(unittest.TestCase):
    """load_thresholds 返回的是「默认叠加已保存」，拿不回原始默认值。"""

    def test_defaults_are_a_separate_payload_field(self):
        source = (Path(__file__).resolve().parents[1] / "kplab" / "server.py").read_text("utf-8")
        self.assertIn('"minimapDefaults": minimap.default_thresholds()', source)

    def test_defaults_are_not_polluted_by_saved_values(self):
        # 同一个调用连取两次必须互不影响 —— 返回的是新字典，不是共享的那一份。
        first = minimap.default_thresholds()
        first["blueMinB"] = 999
        self.assertNotEqual(minimap.default_thresholds()["blueMinB"], 999)


if __name__ == "__main__":
    unittest.main()
