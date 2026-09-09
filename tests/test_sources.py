"""录像来源解析层的安全、隐私、批量和流式处理守卫。"""

from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kplab import server, sources, store, video


class SourceDetection(unittest.TestCase):
    def test_detects_supported_public_sources(self):
        cases = {
            "https://www.bilibili.com/video/BV1xx": "BILIBILI_PAGE",
            "https://v.huya.com/play/123.html": "HUYA_PAGE",
            "https://v.douyu.com/show/abc": "DOUYU_PAGE",
            "https://cdn.example.com/game.mp4?token=secret": "HTTP_VIDEO",
            "https://cdn.example.com/master.m3u8": "HLS",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(sources.detect(value)["sourceType"], expected)

    def test_blocks_local_network_and_credentials(self):
        for value in (
            "http://127.0.0.1/private.mp4", "http://192.168.1.2/a.m3u8",
            "http://localhost/video", "http://user:pass@example.com/video.mp4",
            "file:///etc/passwd", "javascript:alert(1)",
        ):
            with self.subTest(value=value), self.assertRaises(sources.SourceError):
                sources.detect(value)

    def test_multiline_and_csv_are_one_batch(self):
        entries = sources.parse_submission(
            "https://a.example/one.mp4\n# comment\nhttps://b.example/two.m3u8",
            "source_url,team,opponent,event,note\n"
            "https://c.example/three.mp4,AG,狼队,KPL,常规赛",
        )
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[2]["metadata"]["team"], "AG")
        self.assertNotIn("source_url", entries[2]["metadata"])


class ResolverBoundaries(unittest.TestCase):
    def test_login_headers_stay_in_memory_for_ffmpeg(self):
        media_url, scout_url, analysis_url, headers = sources._media_url({
            "url": "https://signed.example/video.m3u8?token=secret",
            "http_headers": {"Cookie": "session=secret", "Referer": "https://example.com"},
        })
        self.assertIn("token=secret", media_url)
        self.assertIsNone(scout_url)
        self.assertIsNone(analysis_url)
        self.assertEqual(headers["Cookie"], "session=secret")

    def test_public_preview_never_contains_media_url_or_headers(self):
        resolved = sources.ResolvedVideoSource(
            platform="bilibili", sourceType="BILIBILI_PAGE",
            originUrl="https://www.bilibili.com/video/BV1",
            sourceUrl="https://signed.cdn.example/video.m4s?token=secret",
            title="test", headers={"Cookie": "secret", "Referer": "example"},
        )
        public = resolved.public()
        self.assertNotIn("sourceUrl", public)
        self.assertNotIn("scoutSourceUrl", public)
        self.assertNotIn("analysisSourceUrl", public)
        self.assertNotIn("headers", public)
        self.assertFalse(public["sourceUrlStored"])
        self.assertNotIn("secret", repr(public))

    def test_resolver_uses_low_quality_for_scout_and_720p_for_analysis(self):
        _media, scout, analysis, _headers = sources._media_url({
            "url": "https://cdn.example/1080.m3u8",
            "formats": [
                {"height": 1080, "url": "https://cdn.example/1080.m3u8"},
                {"height": 360, "url": "https://cdn.example/360.m3u8"},
                {"height": 720, "url": "https://cdn.example/720.m3u8"},
            ],
        })
        self.assertIn("360", scout)
        self.assertIn("720", analysis)

    def test_playlist_keeps_original_indexes(self):
        payload = {"entries": [
            {"id": "p1", "title": "第一局", "url": "https://cdn.example/p1.mp4"},
            {"id": "p2", "title": "第二局", "url": "https://cdn.example/p2.mp4"},
            {"id": "p3", "title": "第三局", "url": "https://cdn.example/p3.mp4"},
        ]}
        with patch.object(sources, "_run_ytdlp", return_value=payload):
            all_items = sources.resolve("https://www.bilibili.com/video/BV1")
            selected = sources.resolve("https://www.bilibili.com/video/BV1", playlist_index=2)
        self.assertEqual([item.playlistIndex for item in all_items], [1, 2, 3])
        self.assertEqual(selected[0].playlistIndex, 2)
        self.assertEqual(selected[0].title, "第二局")

    def test_only_transient_media_failures_are_retryable(self):
        self.assertEqual(sources.classify_media_error("HTTP 404 Not Found"), "VIDEO_NOT_FOUND")
        self.assertEqual(sources.classify_media_error("403 Forbidden"), "ACCESS_DENIED")
        self.assertEqual(sources.classify_media_error("signature expired"), "MEDIA_URL_EXPIRED")
        self.assertEqual(sources.classify_media_error("connection reset"), "NETWORK_ERROR")


class SourcePersistence(unittest.TestCase):
    def test_duplicate_fingerprint_and_batch_round_trip(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            store.create_game(data_dir, "G1", {"mediaFingerprint": "abc"})
            self.assertEqual(sources.find_duplicate(data_dir, "abc"), "G1")
            batch = {"batchId": "SOURCE_1", "createdAt": "now", "jobs": [
                {"status": "PENDING"}, {"status": "DONE"}, {"status": "FAILED"},
            ]}
            sources.save_batch(data_dir, batch)
            self.assertEqual(sources.load_batch(data_dir, "SOURCE_1"), batch)
            summary = sources.list_batches(data_dir)[0]
            self.assertEqual((summary["pending"], summary["done"], summary["failed"]),
                             (1, 1, 1))
            self.assertIsNone(summary["active"])

    def test_batch_summary_exposes_safe_active_progress(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            sources.save_batch(data_dir, {"batchId": "SOURCE_ACTIVE", "jobs": [{
                "gameId": "G1", "title": "第一场", "status": "RUNNING",
                "stage": "STREAMING", "progress": 0.25,
                "currentVideoTime": 123, "updatedAt": "now",
                "originUrl": "https://secret.example/video.m3u8?token=secret",
            }]})
            active = sources.list_batches(data_dir)[0]["active"]
            self.assertEqual(active["progress"], 0.25)
            self.assertEqual(active["currentVideoTime"], 123)
            self.assertNotIn("originUrl", active)
            self.assertNotIn("secret", repr(active))

    def test_cache_has_age_and_size_limits(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            cache = data_dir / "kpl" / "cache"
            cache.mkdir(parents=True)
            old = cache / "old.part"
            old.write_bytes(b"x" * 20)
            old_time = time.time() - 48 * 3600
            old.touch()
            import os
            os.utime(old, (old_time, old_time))
            current = cache / "current.part"
            current.write_bytes(b"y" * 20)
            report = sources.cleanup_cache(data_dir, retention_hours=24, max_bytes=30)
            self.assertFalse(old.exists())
            self.assertTrue(current.exists())
            self.assertEqual(report["removedFiles"], 1)

    def test_interrupted_running_job_returns_to_recovery_queue(self):
        with tempfile.TemporaryDirectory() as raw:
            data_dir = Path(raw)
            batch = {"batchId": "SOURCE_INTERRUPTED", "jobs": [
                {"status": "RUNNING", "stage": "STREAMING"},
                {"status": "DONE", "stage": "COMPLETE"},
            ]}
            sources.save_batch(data_dir, batch)
            found = sources.interrupted_batches(data_dir)
            self.assertEqual([item["batchId"] for item in found], ["SOURCE_INTERRUPTED"])
            self.assertEqual(sources.prepare_interrupted_batch(data_dir, found[0]), 1)
            saved = sources.load_batch(data_dir, "SOURCE_INTERRUPTED")
            self.assertEqual(saved["jobs"][0]["status"], "PENDING")
            self.assertEqual(saved["jobs"][0]["stage"], "RECOVERY_QUEUED")
            self.assertEqual(saved["jobs"][1]["status"], "DONE")


class SourceApi(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        server.CONSOLE = server.Console(self.data_dir)

    def tearDown(self):
        server.CONSOLE = None
        self.temporary.cleanup()

    def test_preview_keeps_good_item_when_another_url_fails(self):
        good = sources.ResolvedVideoSource(
            platform="direct", sourceType="HTTP_VIDEO",
            originUrl="https://cdn.example/good.mp4",
            sourceUrl="https://signed.example/good.mp4?token=secret",
            title="good", mediaFingerprint="fp-good", resolverName="direct-media",
        )
        def resolve_one(origin, *_args, **_kwargs):
            if "missing" in origin:
                raise sources.SourceError("VIDEO_NOT_FOUND")
            return [good]

        with patch.object(sources, "resolve", side_effect=resolve_one):
            result = server.api_source_preview({
                "text": "https://cdn.example/good.mp4\nhttps://cdn.example/missing.mp4"
            })
        self.assertTrue(result["ok"])
        self.assertEqual((result["resolved"], result["failed"]), (1, 1))
        self.assertNotIn("sourceUrl", result["items"][0])
        self.assertNotIn("secret", repr(result))

    def test_created_batch_persists_origin_but_never_media_url(self):
        item = {
            "status": "READY", "originUrl": "https://cdn.example/game.mp4",
            "platform": "direct", "sourceType": "HTTP_VIDEO", "title": "game",
            "mediaFingerprint": "fp-one", "resolverName": "direct-media",
            "metadata": {},
        }
        with patch.object(server.CONSOLE, "start_job", return_value={"started": True}):
            result = server.api_source_start({"items": [item], "gamePatch": "S44"})
        self.assertTrue(result["started"])
        batch = sources.load_batch(self.data_dir, result["batchId"])
        self.assertIsNotNone(batch)
        self.assertIn("originUrl", batch["jobs"][0])
        self.assertNotIn("sourceUrl", batch["jobs"][0])
        self.assertEqual(batch["processing"]["samplingMode"], "smart")
        meta = store.load_meta(self.data_dir, result["created"][0]["gameId"])
        self.assertFalse(meta["sourceUrlStored"])
        self.assertEqual(meta["videoImportMode"], "remote-stream-no-full-copy")

    def test_processing_prefers_720p_analysis_stream(self):
        game_id = "ANALYSIS_STREAM_001"
        store.create_game(self.data_dir, game_id, {"title": "高清录像"})
        batch = {"batchId": "SOURCE_ANALYSIS", "jobs": [{
            "gameId": game_id, "originUrl": "https://example.com/game",
            "title": "高清录像", "status": "PENDING", "stage": "QUEUED",
        }]}
        resolved = sources.ResolvedVideoSource(
            platform="direct", sourceType="HTTP_VIDEO", originUrl="https://example.com/game",
            sourceUrl="https://cdn.example/1440.m3u8",
            analysisSourceUrl="https://cdn.example/720.m3u8", title="高清录像",
            resolvedAt="2026-09-09T00:00:00+1000",
        )
        with patch.object(sources, "resolve", return_value=[resolved]), \
                patch.object(video, "probe_input", return_value=video.VideoInfo(
                    resolved.analysisSourceUrl, 10, 1280, 720, 60)) as probe, \
                patch.object(video, "stream_extract_frames", return_value=[
                    {"index": 0, "atSec": 0, "file": "stream.jpg"}]) as extract:
            server._source_batches_work(self.data_dir, [batch])(lambda _message: None)
        self.assertEqual(probe.call_args.args[0], resolved.analysisSourceUrl)
        self.assertEqual(extract.call_args.args[0], resolved.analysisSourceUrl)

    def test_resumed_progress_includes_already_finished_games(self):
        batch = {"batchId": "SOURCE_PROGRESS", "jobs": [
            {"gameId": "DONE_1", "title": "已完成", "status": "DONE", "stage": "COMPLETE"},
            {"gameId": "NEXT_2", "originUrl": "https://example.com/next.mp4",
             "title": "待处理", "status": "PENDING", "stage": "QUEUED"},
        ]}
        store.create_game(self.data_dir, "NEXT_2", {"title": "待处理"})
        resolved = sources.ResolvedVideoSource(
            platform="direct", sourceType="HTTP_VIDEO",
            originUrl="https://example.com/next.mp4", sourceUrl="https://example.com/next.mp4",
            title="待处理", resolvedAt="2026-09-09T00:00:00+1000",
        )
        observed: list[tuple[float, int]] = []
        original_progress = server.CONSOLE.progress

        def capture_progress(done=None, total=None, **details):
            original_progress(done, total, **details)
            if done is not None and total is not None:
                observed.append((done, total))

        with patch.object(server.CONSOLE, "progress", side_effect=capture_progress), \
                patch.object(sources, "resolve", return_value=[resolved]), \
                patch.object(video, "probe_input", return_value=video.VideoInfo(
                    resolved.sourceUrl, 10, 1280, 720, 60)), \
                patch.object(video, "stream_extract_frames", return_value=[
                    {"index": 0, "atSec": 0, "file": "stream.jpg"}]):
            server._source_batches_work(self.data_dir, [batch])(lambda _message: None)
        self.assertEqual(observed[0], (1, 2))
        self.assertEqual(observed[-1], (2, 2))

    def test_existing_interrupted_batch_can_resume_without_resubmitting_url(self):
        batch = {"batchId": "SOURCE_RESUME", "cookieBrowser": "chrome", "jobs": [{
            "gameId": "G1", "originUrl": "https://www.huya.com/video/play/1.html",
            "title": "第一场", "status": "RUNNING", "stage": "RESOLVING",
        }]}
        sources.save_batch(self.data_dir, batch)
        with patch.object(server.CONSOLE, "start_job", return_value={"started": True}):
            result = server.api_source_resume({"batchId": "SOURCE_RESUME"})
        self.assertTrue(result["started"])
        saved = sources.load_batch(self.data_dir, "SOURCE_RESUME")
        self.assertEqual(saved["jobs"][0]["status"], "PENDING")
        self.assertTrue(saved["jobs"][0]["recoveredAfterRestart"])

    def test_long_vod_becomes_scout_job_instead_of_failing(self):
        game_id = "LONG_VOD_001"
        store.create_game(self.data_dir, game_id, {"title": "长回放"})
        batch = {
            "batchId": "SOURCE_LONG", "cookieBrowser": None,
            "processing": {"samplingMode": "smart", "everySec": 5,
                           "startSec": 0, "endSec": None},
            "jobs": [{"gameId": game_id, "originUrl": "https://example.com/long.mp4",
                      "title": "长回放", "status": "PENDING", "stage": "QUEUED"}],
        }
        sources.save_batch(self.data_dir, batch)
        resolved = sources.ResolvedVideoSource(
            platform="direct", sourceType="HTTP_VIDEO",
            originUrl="https://example.com/long.mp4",
            sourceUrl="https://cdn.example/long.mp4", title="长回放",
            resolvedAt="2026-09-09T00:00:00+1000",
        )

        with patch.object(sources, "resolve", return_value=[resolved]), \
                patch.object(video, "probe_input", return_value=video.VideoInfo(
                    resolved.sourceUrl, 9393, 1920, 1080, 60)), \
                patch.object(video, "stream_extract_frames", side_effect=video.VideoError(
                    "这段录像会超过2000张基础帧，请缩短范围")), \
                patch.object(video, "sparse_seek_frames", return_value=[
                    {"index": 0, "atSec": 0.0, "file": "scout_000000.000s.jpg"}]):
            server._source_batches_work(self.data_dir, [batch])(lambda _message: None)
        saved = sources.load_batch(self.data_dir, "SOURCE_LONG")
        self.assertEqual(saved["jobs"][0]["status"], "SCOUT_DONE")
        meta = store.load_meta(self.data_dir, game_id)
        self.assertTrue(meta["sourceSegmentationRequired"])
        self.assertEqual(meta["samplingMode"], "long-source-scout")

    def test_public_page_retry_drops_browser_cookie_after_timeout(self):
        game_id = "COOKIE_RETRY_001"
        store.create_game(self.data_dir, game_id, {"title": "公开录像"})
        batch = {
            "batchId": "SOURCE_COOKIE", "cookieBrowser": "chrome",
            "jobs": [{"gameId": game_id, "originUrl": "https://example.com/game.mp4",
                      "title": "公开录像", "status": "PENDING", "stage": "QUEUED"}],
        }
        resolved = sources.ResolvedVideoSource(
            platform="direct", sourceType="HTTP_VIDEO",
            originUrl="https://example.com/game.mp4",
            sourceUrl="https://cdn.example/game.mp4", title="公开录像",
        )
        with patch.object(sources, "resolve", side_effect=[
                sources.SourceError("NETWORK_ERROR"), [resolved]]) as resolver, \
                patch.object(video, "probe_input", return_value=video.VideoInfo(
                    resolved.sourceUrl, 10, 1920, 1080, 60)), \
                patch.object(video, "stream_extract_frames", return_value=[
                    {"index": 0, "atSec": 0.0, "file": "stream_000000.000s.jpg"}]):
            server._source_batches_work(self.data_dir, [batch])(lambda _message: None)
        self.assertEqual(resolver.call_args_list[0].args[1], "chrome")
        self.assertEqual(resolver.call_args_list[1].args[1], "")

    def test_scout_segments_create_independent_game_records(self):
        parent_game = "LONG_PARENT_001"
        store.create_game(self.data_dir, parent_game, {
            "title": "直播回放", "sampleType": "pro_player_ranked",
            "sourceSegments": [
                {"startSec": 180, "endSec": 1080, "status": "READY"},
                {"startSec": 1260, "endSec": 2220, "status": "READY"},
            ],
        })
        sources.save_batch(self.data_dir, {
            "batchId": "SOURCE_PARENT", "cookieBrowser": None,
            "jobs": [{"gameId": parent_game,
                      "originUrl": "https://www.huya.com/video/play/1.html",
                      "platform": "huya", "sourceType": "HUYA_PAGE"}],
        })
        with patch.object(server.CONSOLE, "start_job", return_value={"started": True}):
            result = server.api_source_segments_start({
                "batchId": "SOURCE_PARENT", "parentGameId": parent_game,
            })
        self.assertTrue(result["started"])
        self.assertEqual(result["count"], 2)
        child = store.load_meta(self.data_dir, result["created"][0]["gameId"])
        self.assertEqual(child["sourceParentGameId"], parent_game)
        self.assertEqual(child["segmentDurationSec"], 900)
        self.assertFalse(child["segmentBoundaryIsTrainingTruth"])
        parent = store.load_meta(self.data_dir, parent_game)
        self.assertEqual(parent["analysisStatus"], "SEGMENT_EXTRACTION_RUNNING")


class StreamingDecoder(unittest.TestCase):
    def test_hls_scout_maps_video_time_to_small_media_segments(self):
        manifest = ("#EXTM3U\n#EXT-X-TARGETDURATION:5\n"
                    "#EXTINF:5.0,\npart/000.ts?sig=x\n"
                    "#EXTINF:5.0,\npart/001.ts?sig=x\n"
                    "#EXTINF:5.0,\npart/002.ts?sig=x\n")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _maximum):
                return manifest.encode()

        with patch.object(video, "urlopen", return_value=Response()):
            targets = video._hls_scout_targets(
                "https://cdn.example/path/master.m3u8?token=secret", {}, [0, 6, 11]
            )
        self.assertEqual([item[3] for item in targets], [0.0, 0.0, 0.0])
        self.assertIn("part/001.ts", targets[1][2])

    def test_dispatcher_gives_each_consumer_its_own_rate(self):
        fast: list[float] = []
        slow: list[float] = []
        dispatcher = video.FrameDispatcher()
        dispatcher.subscribe("fast", 0.5, lambda at, frame: fast.append(at))
        dispatcher.subscribe("slow", 1.0, lambda at, frame: slow.append(at))
        for at in (0.0, 0.25, 0.5, 0.75, 1.0):
            dispatcher.dispatch(at, b"frame")
        self.assertEqual(fast, [0.0, 0.5, 1.0])
        self.assertEqual(slow, [0.0, 1.0])

    def test_one_ffmpeg_process_writes_only_sparse_jpegs(self):
        first = b"\xff\xd8one\xff\xd9"
        second = b"\xff\xd8two\xff\xd9"

        class FakeProcess:
            def __init__(self):
                self.stdout = io.BytesIO(first + second)
                self.stderr = io.BytesIO(b"")
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                self.returncode = -15

        with tempfile.TemporaryDirectory() as raw, \
                patch.object(video, "ffmpeg_path", return_value="ffmpeg"), \
                patch.object(video.subprocess, "Popen", return_value=FakeProcess()) as popen:
            records = video.stream_extract_frames(
                "https://cdn.example/game.mp4", Path(raw), 9, mode="smart"
            )
            self.assertEqual(popen.call_count, 1, "远程录像只能连续解码一次")
            self.assertEqual(len(records), 2)
            self.assertEqual(len(list(Path(raw).glob("*.jpg"))), 2)
            self.assertFalse(list(Path(raw).glob("*.mp4")), "不应留下完整录像副本")

    def test_header_newlines_cannot_inject_extra_headers(self):
        value = video._header_argument({"Referer\nInjected": "ok\r\nX-Evil: yes"})
        self.assertNotIn("\nInjected", value)
        self.assertNotIn("\nX-Evil", value)

    def test_saved_error_never_contains_signed_media_url(self):
        message = sources.safe_error(
            "HTTP 403 https://cdn.example/video.m3u8?token=topsecret Cookie: abc"
        )
        self.assertNotIn("topsecret", message)
        self.assertNotIn("abc", message)

    def test_long_source_limit_is_not_reported_as_network_failure(self):
        self.assertEqual(
            sources.classify_media_error("这段录像会超过2000张基础帧，请缩短范围"),
            "SOURCE_TOO_LONG",
        )


if __name__ == "__main__":
    unittest.main()
