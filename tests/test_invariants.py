"""锁住三条不许违反的性质。

这个文件存在的理由：本项目最危险的错误**都不会报错**。
把「没读到」写成 0、用后面的帧修前面的帧、把黑暗暴君算进暴君 ——
这三种改动都能让代码正常跑、让报表更好看，只是把数据悄悄弄坏。

所以它们必须被测试钉住。任何一条测试挂了，
都不要「顺手改一下测试让它通过」—— 那说明改动本身是错的。

跑法（不需要装任何东西）：
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kplab import annotate, check, evaluate, ocr, paths, rules, samples, schema, season_s44, state, store, video  # noqa: E402


def ocr_field(value, confidence=0.9):
    return schema.field(value, schema.SOURCE_OCR, confidence)


class NoFutureLeakage(unittest.TestCase):
    """性质一：t 时刻的 State 只能含 t 时刻及之前的信息。"""

    def test_bad_later_read_does_not_rewrite_earlier_frame(self):
        observations = [
            {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}},
            {"atSec": 120, "fields": {"blueGold": ocr_field(2000)}},   # 倒退，必是错的
            {"atSec": 180, "fields": {"blueGold": ocr_field(4200)}},
        ]
        built = state.build({"gameId": "T"}, observations, rules.load())
        first = built["frames"][0]["teams"]["100"]["totalGold"]
        self.assertEqual(schema.get(first), 3000,
                         "第 2 帧的坏读数回头改了第 1 帧 —— 这是未来信息泄漏")

    def test_contradicting_read_is_rejected_not_accepted(self):
        observations = [
            {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}},
            {"atSec": 120, "fields": {"blueGold": ocr_field(2000)}},
        ]
        built = state.build({"gameId": "T"}, observations, rules.load())
        second = built["frames"][1]["teams"]["100"]["totalGold"]
        self.assertEqual(schema.src(second), schema.SOURCE_CARRIED,
                         "与已知的过去矛盾的读数应该被丢弃并沿用旧值")
        self.assertTrue(built["rejections"], "拒绝了读数就该有记录，否则没人知道发生过")

    def test_observations_are_processed_in_time_order(self):
        # 乱序输入也必须按时间处理，否则「上一帧」就不是时间上更早的那一帧
        observations = [
            {"atSec": 180, "fields": {"blueGold": ocr_field(4200)}},
            {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}},
            {"atSec": 120, "fields": {"blueGold": ocr_field(3800)}},
        ]
        built = state.build({"gameId": "T"}, observations, rules.load())
        times = [f["atSec"] for f in built["frames"]]
        self.assertEqual(times, sorted(times))
        golds = [schema.get(f["teams"]["100"]["totalGold"]) for f in built["frames"]]
        self.assertEqual(golds, [3000, 3800, 4200])

    def test_carried_confidence_decays(self):
        """沿用越久越不可信，最终必须跌破可信线，不能一直冒充事实。"""
        observations = [{"atSec": 60, "fields": {"blueGold": ocr_field(3000, 0.9)}}]
        observations += [{"atSec": 60 + 30 * i, "fields": {}} for i in range(1, 6)]
        built = state.build({"gameId": "T"}, observations, rules.load())
        confidences = [schema.conf(f["teams"]["100"]["totalGold"]) for f in built["frames"]]
        self.assertTrue(all(b <= a for a, b in zip(confidences, confidences[1:])),
                        "沿用值的置信度必须单调不增")
        self.assertLess(confidences[-1], schema.TRUST_THRESHOLD,
                        "沿用太久的值还被当作可信 —— 这会让陈旧数据冒充事实")


class ObjectiveCountersStaySeparate(unittest.TestCase):
    """性质二：五种中立资源绝不合并（LOL ③ 号 bug 的教训）。"""

    def test_five_objectives_have_five_counters(self):
        observations = [{
            "atSec": 600,
            "fields": {},
            "events": [
                {"type": "TYRANT_KILL", "teamId": 100},
                {"type": "DARK_TYRANT_KILL", "teamId": 100},
                {"type": "OVERLORD_KILL", "teamId": 100},
                {"type": "STORM_DRAGON_KILL", "teamId": 100},
                {"type": "PROPHET_OVERLORD_KILL", "teamId": 100},
            ],
        }]
        built = state.build({"gameId": "T"}, observations, rules.load())
        team = built["frames"][0]["teams"]["100"]
        for key in ("tyrants", "darkTyrants", "overlords", "stormDragons", "prophetOverlords"):
            self.assertEqual(schema.get(team[key]), 1,
                             f"{key} 应该正好是 1 —— 不是 0（漏了）也不是 2（和别的合并了）")

    def test_dark_tyrant_is_not_a_tyrant(self):
        observations = [{"atSec": 600, "fields": {},
                         "events": [{"type": "DARK_TYRANT_KILL", "teamId": 200}]}]
        built = state.build({"gameId": "T"}, observations, rules.load())
        team = built["frames"][0]["teams"]["200"]
        self.assertEqual(schema.get(team["darkTyrants"]), 1)
        self.assertEqual(schema.get(team["tyrants"]), 0,
                         "黑暗暴君被算进了暴君 —— 正是 LOL 那个 bug 的翻版")

    def test_storm_dragon_is_not_an_overlord(self):
        observations = [{"atSec": 900, "fields": {},
                         "events": [{"type": "STORM_DRAGON_KILL", "teamId": 100}]}]
        built = state.build({"gameId": "T"}, observations, rules.load())
        team = built["frames"][0]["teams"]["100"]
        self.assertEqual(schema.get(team["stormDragons"]), 1)
        self.assertEqual(schema.get(team["overlords"]), 0,
                         "风暴龙王被算进了主宰 —— 两者价值差一个数量级")

    def test_kill_credits_the_right_side(self):
        observations = [{"atSec": 100, "fields": {},
                         "events": [{"type": "HERO_KILL", "teamId": 100}]}]
        built = state.build({"gameId": "T"}, observations, rules.load())
        frame = built["frames"][0]
        self.assertEqual(schema.get(frame["teams"]["100"]["kills"]), 1)
        self.assertEqual(schema.get(frame["teams"]["200"]["deaths"]), 1)


class MissingIsNotZero(unittest.TestCase):
    """性质三：读不出来是「不知道」，绝不是 0。"""

    def test_unknown_field_stays_none(self):
        built = state.build({"gameId": "T"},
                            [{"atSec": 60, "fields": {"blueGold": ocr_field(3000)}}],
                            rules.load())
        red = built["frames"][0]["teams"]["200"]["totalGold"]
        self.assertIsNone(schema.get(red), "没读到的经济被写成了 0")

    def test_derived_value_is_unknown_when_any_input_is_unknown(self):
        built = state.build({"gameId": "T"},
                            [{"atSec": 60, "fields": {"blueGold": ocr_field(3000)}}],
                            rules.load())
        diff = built["frames"][0]["diff"]["totalGold"]
        self.assertIsNone(schema.get(diff),
                          "红方经济未知时，经济差必须是未知 —— "
                          "否则会凭空造出 +3000 的假优势")

    def test_derive_itself_returns_unknown_when_any_input_is_unknown(self):
        """直接盯住 schema.derive 的契约，不要只靠调用方的前置判断。

        state.py 在调用 derive 之前自己也检查了一遍，所以即使 derive 的守卫被删掉，
        经济差那条测试照样能过 —— 这正是一次变异测试暴露出来的盲点。
        契约要在契约所在的那一层测，否则下一个调用方就会掉进去。
        """
        known = ocr_field(100)
        self.assertIsNone(schema.get(schema.derive(20, known, schema.unknown())),
                          "只要有一个输入未知，派生值就必须是未知")
        self.assertIsNone(schema.get(schema.derive(20, schema.unknown())))
        self.assertEqual(schema.src(schema.derive(20, known, schema.unknown())),
                         schema.SOURCE_UNKNOWN)

    def test_derived_confidence_is_the_weakest_input(self):
        low, high = ocr_field(100, 0.4), schema.field(80, schema.SOURCE_MANUAL, 0.98)
        self.assertAlmostEqual(schema.conf(schema.derive(20, low, high)), 0.4,
                               msg="做一次减法不该让结果变得更确定")

    def test_field_with_none_value_cannot_claim_a_source(self):
        entry = schema.field(None, schema.SOURCE_MANUAL, 1.0)
        self.assertEqual(schema.src(entry), schema.SOURCE_UNKNOWN)
        self.assertEqual(schema.conf(entry), 0.0)

    def test_ocr_returns_none_not_zero_on_failure(self):
        for text in ("", "abc", "———", "99999999"):
            value, confidence = ocr.parse_int(text, lo=0, hi=200000)
            self.assertIsNone(value, f"{text!r} 解析失败时应返回 None 而不是 0")
            self.assertEqual(confidence, 0.0)


class RuleValidation(unittest.TestCase):
    """规则校验：不可能的值必须挡在写入之前，而不是留到报告里。"""

    def test_out_of_range_level_is_rejected_at_annotation(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {"players": {"1": {"level": 99}}})

    def test_pixel_coordinates_are_rejected(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {"players": {"1": {"x": 420}}})

    def test_unknown_event_type_is_rejected(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {"events": [{"type": "BARON", "teamId": 100}]})

    def test_empty_annotation_is_rejected(self):
        with self.assertRaises(annotate.AnnotateError):
            annotate.build_observation(120, {})

    def test_clock_needs_two_digit_seconds(self):
        self.assertEqual(ocr.parse_clock("12:34")[0], 754)
        self.assertEqual(ocr.parse_clock("1 2 : 3 4")[0], 754)   # OCR 塞进来的空格
        self.assertIsNone(ocr.parse_clock("2.3万")[0],
                          "把经济读成了时间 —— 会让整场时间轴错位")

    def test_game_id_blocks_path_traversal(self):
        self.assertFalse(store.valid_game_id("../evil"))
        self.assertFalse(store.valid_game_id("a/b"))
        self.assertTrue(store.valid_game_id("KPL2026_AG_vs_EST_G1"))


class StorageRoundTrip(unittest.TestCase):
    """存储：压缩 + 只追加，读回来要和写进去的一样。"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_append_only_jsonl_reads_back_in_order(self):
        path = store.observations_path(self.dir, "G1")
        for i in range(5):
            store.append_jsonl_gz(path, {"i": i})
        self.assertEqual([r["i"] for r in store.read_jsonl_gz(path)], list(range(5)))

    def test_files_are_actually_compressed(self):
        import gzip
        path = store.state_path(self.dir, "G1")
        store.write_json_gz(path, {"x": "重复内容" * 500})
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            self.assertIn("重复内容", handle.read())
        self.assertLess(path.stat().st_size, 1000, "看起来没有真的压缩")

    def test_annotation_is_written_to_both_streams(self):
        store.create_game(self.dir, "G1", {})
        observation = annotate.build_observation(60, {"blueGold": 3000})
        annotate.save(self.dir, "G1", observation)
        # annotations 是纯人工的标尺，observations 是喂给 state 的输入，两边都要有
        self.assertEqual(len(annotate.load_annotations(self.dir, "G1")), 1)
        self.assertEqual(
            len(list(store.read_jsonl_gz(store.observations_path(self.dir, "G1")))), 1)


class EvaluationHonesty(unittest.TestCase):
    """评估：没有标尺时必须说「量不了」，不能给一个准确率。"""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        store.create_game(self.dir, "G1", {})

    def test_no_annotations_means_no_accuracy_number(self):
        store.append_jsonl_gz(store.observations_path(self.dir, "G1"),
                              {"atSec": 60, "fields": {"blueGold": ocr_field(3000)}})
        report = evaluate.run(self.dir, "G1")
        self.assertFalse(report["verdict"]["ok"])
        self.assertIsNone(report.get("exactRate"),
                          "没有人工标注却给出了准确率 —— 那个数字是编的")

    def test_detects_a_wrong_ocr_read(self):
        annotate.save(self.dir, "G1", annotate.build_observation(60, {"blueGold": 11000}))
        store.append_jsonl_gz(store.observations_path(self.dir, "G1"),
                              {"atSec": 60, "fields": {"blueGold": ocr_field(1100)}})
        report = evaluate.run(self.dir, "G1")
        self.assertEqual(report["exactRate"], 0.0)
        self.assertFalse(report["verdict"]["ok"])


class RulesAreMarkedUnverified(unittest.TestCase):
    """已提供的S44常量和仍未知的常量必须明确分开。"""

    def test_only_user_provided_s44_values_are_confirmed(self):
        table = rules.load()
        self.assertEqual(rules.value(table, "tyrantFirstSpawnSec"), 240)
        self.assertEqual(rules.value(table, "overlordFirstSpawnSec"), 240)
        self.assertEqual(rules.value(table, "stormDragonFromSec"), 1200)
        self.assertTrue(table["tyrantFirstSpawnSec"]["verified"])
        self.assertEqual(table["tyrantFirstSpawnSec"]["season"], "S44")
        self.assertTrue(table["tyrantFirstSpawnSec"]["official"])
        self.assertFalse(table["tyrantFirstSpawnSec"]["externalVerified"])
        self.assertIn("maxLevel", rules.unverified(table),
                      "用户没有在本批信息中确认最高等级，不能顺手标成已核对")

    def test_override_marks_verified(self):
        directory = Path(tempfile.mkdtemp())
        rules.save_override(directory, "towersPerLane", 2, "测试")
        table = rules.load(directory)
        self.assertTrue(table["towersPerLane"]["verified"])
        self.assertNotIn("towersPerLane", rules.unverified(table))

    def test_s44_knowledge_keeps_ambiguities_out_of_calculation(self):
        payload = season_s44.payload()
        self.assertEqual(payload["season"], "S44")
        self.assertTrue(payload["official"])
        self.assertEqual(payload["officialStatus"], "user_attested_official")
        self.assertFalse(payload["externalVerified"])
        ambiguous = {entry["id"]: entry for entry in payload["rules"]
                     if not entry["machineActive"]}
        ids = [entry["id"] for entry in payload["rules"]]
        self.assertEqual(len(ids), len(set(ids)), "赛季规则编号重复会导致覆盖或错误引用")
        self.assertIn("minion_speed", ambiguous)
        self.assertIn("red_falcon", ambiguous)
        self.assertGreater(payload["machineActiveCount"], 15)

    def test_official_ambiguities_are_not_guessed(self):
        payload = season_s44.payload()
        entries = {entry["id"]: entry for entry in payload["rules"]}
        self.assertEqual(entries["minion_speed"]["values"]["sourceSideEndText"],
                         "180分钟")
        self.assertFalse(entries["minion_speed"]["machineActive"])
        self.assertEqual(entries["red_falcon"]["values"]["teamGoldSourceText"],
                         "25～20")
        self.assertFalse(entries["red_falcon"]["machineActive"])

    def test_s44_spawn_times_reject_impossible_early_kills(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            store.create_game(data_dir, "S44_BAD", {})
            for observation in (
                {"atSec": 20, "fields": {},
                 "events": [{"type": "RED_BUFF_KILL", "teamId": 100}]},
                {"atSec": 500, "fields": {},
                 "events": [{"type": "DARK_TYRANT_KILL", "teamId": 100}]},
                {"atSec": 1100, "fields": {},
                 "events": [{"type": "STORM_DRAGON_KILL", "teamId": 200}]},
            ):
                store.append_jsonl_gz(store.observations_path(data_dir, "S44_BAD"), observation)
            report = check.run(data_dir, "S44_BAD")
            descriptions = " ".join(item["what"] for item in report["ruleViolations"])
            self.assertIn("红蓝石像", descriptions)
            self.assertIn("暗影暴君", descriptions)
            self.assertIn("风暴龙王", descriptions)


class DataDirectoryDiscovery(unittest.TestCase):
    """数据目录只能在明确的项目/用户范围内查找，绝不能猜中系统目录。"""

    def test_falls_back_to_project_data_not_ancestor_var(self):
        with tempfile.TemporaryDirectory() as raw:
            outer = Path(raw)
            (outer / "var").mkdir()
            repo = outer / "repo"
            cwd = repo / "tools"
            (repo / ".git").mkdir(parents=True)
            cwd.mkdir()
            with patch.dict("os.environ", {"KPLAB_DATA": ""}), \
                    patch.object(paths.Path, "cwd", return_value=cwd):
                self.assertEqual(paths.find_data_dir(), (repo / "data").resolve())

    def test_existing_project_data_is_found_from_subdirectory(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            cwd = repo / "kplab" / "web"
            (repo / ".git").mkdir(parents=True)
            (repo / "data").mkdir()
            cwd.mkdir(parents=True)
            with patch.dict("os.environ", {"KPLAB_DATA": ""}), \
                    patch.object(paths.Path, "cwd", return_value=cwd):
                self.assertEqual(paths.find_data_dir(), (repo / "data").resolve())

    def test_explicit_path_still_has_highest_priority(self):
        with tempfile.TemporaryDirectory() as raw:
            chosen = Path(raw) / "chosen"
            with patch.dict("os.environ", {"KPLAB_DATA": "/should/not/win"}):
                self.assertEqual(paths.find_data_dir(str(chosen)), chosen.resolve())


class BrowserFrameIngestion(unittest.TestCase):
    """网页只上传抽出的帧，服务端必须把它当图片而不是任意文件。"""

    JPEG = b"\xff\xd8\xff\xe0browser-generated-frame\xff\xd9"

    def test_saves_one_browser_frame_with_parseable_timestamp(self):
        with tempfile.TemporaryDirectory() as raw:
            result = video.save_browser_frame(Path(raw), 60.125, self.JPEG, "image/jpeg")
            saved = Path(raw) / result["file"]
            self.assertTrue(saved.is_file())
            self.assertEqual(saved.read_bytes(), self.JPEG)
            self.assertTrue(saved.name.endswith("60.125s.jpg"))

    def test_rejects_non_image_content_even_with_image_mime(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(video.VideoError):
                video.save_browser_frame(Path(raw), 30, b"not an image", "image/jpeg")
            self.assertEqual(list(Path(raw).iterdir()), [])

    def test_rejects_invalid_or_unbounded_timestamp(self):
        with tempfile.TemporaryDirectory() as raw:
            for value in (-1, float("nan"), 24 * 60 * 60 + 1):
                with self.assertRaises(video.VideoError):
                    video.save_browser_frame(Path(raw), value, self.JPEG, "image/jpeg")


class SampleLibraryAndSampling(unittest.TestCase):
    """样本分层和采样密度属于数据契约，不能只存在于页面文字里。"""

    def test_professional_match_keeps_separate_label_and_weight(self):
        meta = samples.normalize_metadata({
            "sampleType": "professional_match", "focalTeam": "AG",
            "tags": "季后赛，逆风局,季后赛",
        })
        self.assertTrue(meta["isProfessionalMatch"])
        self.assertEqual(meta["sampleTypeLabel"], "职业正式比赛")
        self.assertEqual(meta["trainingWeight"], 1.5)
        self.assertEqual(meta["tags"], ["季后赛", "逆风局"])

    def test_ranked_player_sample_is_not_professional_match(self):
        meta = samples.normalize_metadata({
            "sampleType": "pro_player_ranked", "focalPlayer": "某选手",
        })
        self.assertFalse(meta["isProfessionalMatch"])
        self.assertEqual(meta["analysisScope"], "选手个人决策")

    def test_invalid_weight_is_rejected(self):
        for value in (0, 6, float("nan"), "不是数字"):
            with self.assertRaises(samples.SampleError):
                samples.normalize_metadata({"sampleType": "other", "trainingWeight": value})

    def test_smart_plan_gets_denser_as_game_progresses(self):
        plan = video.sampling_plan(900, mode="smart")
        times = plan["times"]
        early = [b - a for a, b in zip(times, times[1:]) if a < 230]
        middle = [b - a for a, b in zip(times, times[1:]) if 250 <= a < 580]
        late = [b - a for a, b in zip(times, times[1:]) if a >= 620]
        self.assertTrue(early and middle and late)
        self.assertLessEqual(max(early), 8)
        self.assertLessEqual(max(middle), 5)
        self.assertLessEqual(max(late), 3)
        self.assertTrue(plan["adaptive"])
        scan_gaps = [b - a for a, b in zip(plan["scanTimes"], plan["scanTimes"][1:])]
        self.assertLessEqual(max(scan_gaps), 2,
                             "智能模式若不至少每2秒低清扫描一次，会漏掉短暂事件")

    def test_batch_ids_never_overwrite_existing_game(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            store.create_game(data_dir, "KPL_BATCH_001", {"kept": True})
            allocated = store.next_game_id(data_dir, "KPL_BATCH", 1)
            self.assertEqual(allocated, "KPL_BATCH_001_2")
            self.assertTrue(store.load_meta(data_dir, "KPL_BATCH_001")["kept"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class StillImageCropping(unittest.TestCase):
    """裁静态图片时不能加 -ss。

    这条来自一个静默失败的真 bug：`ffmpeg -ss 0 -i 静态图` 什么都不输出，
    但**退出码是 0、stderr 是空的**。调用方只看到「没有文件」，
    完全不知道为什么。

    影响面比发现它的地方大：`ocr.read_regions()` 也是拿帧图 + 0.0 调
    `crop_region` 的，也就是说对帧图跑 OCR 一直拿不到裁剪结果。
    """

    @unittest.skipUnless(__import__("shutil").which("ffmpeg"), "需要 ffmpeg")
    def test_crop_at_zero_seconds_produces_a_file(self):
        import subprocess
        import tempfile
        from kplab import video

        work = Path(tempfile.mkdtemp())
        source = work / "still.png"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "color=c=blue:s=320x180", "-frames:v", "1", str(source)],
            check=True)

        for suffix in (".png", ".jpg"):
            out = work / f"crop{suffix}"
            video.crop_region(source, out, 0.0, (0.1, 0.1, 0.5, 0.5), scale_width=200)
            self.assertTrue(out.is_file(), f"t=0 裁 {suffix} 没有产出文件")
            self.assertGreater(out.stat().st_size, 0)

    def test_failure_message_is_not_empty(self):
        """ffmpeg 静默失败时，错误信息不能也是空的。"""
        import tempfile
        from kplab import video

        work = Path(tempfile.mkdtemp())
        with self.assertRaises(video.VideoError) as caught:
            video.crop_region(work / "nope.png", work / "out.png", 0.0,
                              (0.1, 0.1, 0.5, 0.5))
        self.assertTrue(str(caught.exception).strip(),
                        "抛了异常但没有任何说明，排查时等于没有信息")



class BuiltinHudMatchesHonorOfKingsLayout(unittest.TestCase):
    """内置 HUD 坐标必须符合王者荣耀的布局，不是 LOL 的。

    **这一组来自一个真实的错误。** 第一版把小地图放在了**左下角** ——
    那是 LOL 的习惯（LOL 小地图在右下）。王者荣耀的小地图在**左上角**。
    用户对着真实画面一眼看出不对：裁出来的是赞助商横幅。

    讽刺的是，这个项目的核心文档就叫《为什么王者荣耀不能照抄LOL》，
    而我在坐标上照抄了。所以把方位钉死在测试里。

    这里只钉**方位**（哪个角），不钉具体数值 ——
    具体的框仍然是猜的，必须由人对着真实画面标定。
    """

    @classmethod
    def setUpClass(cls):
        from kplab import hud
        cls.regions = hud.builtin_profiles()["kpl_broadcast"]["regions"]

    def test_minimap_is_top_left_not_bottom(self):
        x, y, w, h = self.regions["minimap"]
        self.assertLess(y, 0.2, "小地图应该在**上**方 —— 放到下面是 LOL 的习惯")
        self.assertLess(x, 0.2, "小地图应该在**左**侧")

    def test_economy_panel_is_bottom_centre(self):
        """用户原话：正式比赛会在屏幕下方中间较长时间展示经济面板。"""
        for key in ("blueGold", "redGold"):
            x, y, w, h = self.regions[key]
            self.assertGreater(y, 0.7, f"{key} 应该在**下**方")
            self.assertTrue(0.2 < x < 0.8, f"{key} 应该靠**中间**")

    def test_clock_and_score_are_top_centre(self):
        for key in ("clock", "blueKills", "redKills"):
            x, y, w, h = self.regions[key]
            self.assertLess(y, 0.2, f"{key} 应该在上方")
            self.assertTrue(0.3 < x < 0.7, f"{key} 应该在中间")

    def test_minimap_does_not_overlap_the_score_area(self):
        """小地图和比分都在上方，但一个靠左一个靠中，不该重叠。"""
        mx, my, mw, mh = self.regions["minimap"]
        cx, cy, cw, ch = self.regions["clock"]
        self.assertLess(mx + mw, cx, "小地图的右边缘伸进了比分区")

    def test_builtin_profile_is_still_marked_uncalibrated(self):
        """方位改对了，但**具体的框仍然是猜的**，不许标成已标定。

        不同录像源的裁剪、分辨率、有没有加边框都不一样。
        """
        from kplab import hud
        profile = hud.builtin_profiles()["kpl_broadcast"]
        self.assertFalse(profile.get("calibrated"),
                         "内置档案被标成已标定了 —— 它只是一组猜测")
        self.assertIn("猜测", profile.get("note", ""))
