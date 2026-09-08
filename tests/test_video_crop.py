import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kplab import video


class CropRegionTests(unittest.TestCase):
    def _run_crop(self, at_sec):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "frame.jpg"
            output = Path(tmp) / "crop.png"
            source.touch()
            commands = []

            def fake_run(command, timeout):
                commands.append(command)
                output.touch()
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(video, "ffmpeg_path", return_value="ffmpeg"), \
                    mock.patch.object(video, "_run", side_effect=fake_run):
                video.crop_region(source, output, at_sec, (0.1, 0.1, 0.2, 0.2), 320)
            return commands[0]

    def test_zero_second_image_crop_does_not_seek(self):
        command = self._run_crop(0.0)
        self.assertNotIn("-ss", command)

    def test_nonzero_video_crop_still_seeks(self):
        command = self._run_crop(12.5)
        index = command.index("-ss")
        self.assertEqual(command[index + 1], "12.500")


if __name__ == "__main__":
    unittest.main()
