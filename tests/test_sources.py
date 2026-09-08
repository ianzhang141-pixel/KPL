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
        media_url, headers = sources._media_url({
            "url": "https://signed.example/video.m3u8?token=secret",
            "http_headers": {"Cookie": "session=secret", "Referer": "https://example.com"},
        })
        self.assertIn("token=secret", media_url)
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
        self.assertNotIn("headers", public)
        self.assertFalse(public["sourceUrlStored"])
        self.assertNotIn("secret", repr(public))

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
        meta = store.load_meta(self.data_dir, result["created"][0]["gameId"])
        self.assertFalse(meta["sourceUrlStored"])
        self.assertEqual(meta["videoImportMode"], "remote-stream-no-full-copy")


class StreamingDecoder(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
