import contextlib
import ctypes
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from algorithm.ppo.file_io import replace_file


ARTIFACTS = Path(__file__).resolve().parents[1] / "result" / "_verification" / "file_io"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


class FileIOTests(unittest.TestCase):
    def test_permission_error_retries_without_deleting_old_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            source, target = Path(directory) / "checkpoint.tmp", Path(directory) / "checkpoint.pt"
            source.write_bytes(b"new-complete-checkpoint")
            target.write_bytes(b"previous-checkpoint")
            original = Path.replace
            attempts = []

            def temporarily_locked(path, destination):
                attempts.append(destination)
                if len(attempts) < 3:
                    self.assertEqual(target.read_bytes(), b"previous-checkpoint")
                    raise PermissionError("simulated Windows sharing conflict")
                return original(path, destination)

            with (
                patch.object(Path, "replace", temporarily_locked),
                patch("algorithm.ppo.file_io.time.sleep") as sleep,
            ):
                self.assertEqual(replace_file(source, target), target)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(target.read_bytes(), b"new-complete-checkpoint")

    def test_persistent_target_lock_saves_full_fallback(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            source, target, fallback = (
                Path(directory) / name
                for name in ("checkpoint.tmp", "checkpoint.pt", "checkpoint_u0085.pt")
            )
            source.write_bytes(b"update-85-complete")
            target.write_bytes(b"update-84-complete")
            original = Path.replace

            def locked_target(path, destination):
                if destination == target:
                    raise PermissionError("target held open")
                return original(path, destination)

            with (
                patch.object(Path, "replace", locked_target),
                patch("algorithm.ppo.file_io.time.sleep") as sleep,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                actual = replace_file(source, target, fallback)
            self.assertEqual(actual, fallback)
            self.assertEqual(sleep.call_count, 7)
            self.assertEqual(target.read_bytes(), b"update-84-complete")
            self.assertEqual(fallback.read_bytes(), b"update-85-complete")
            self.assertFalse(source.exists())

    def test_other_io_errors_are_not_hidden(self):
        source, target = Path("source.tmp"), Path("target.pt")
        with (
            patch.object(Path, "replace", side_effect=OSError("disk write failure")),
            patch("algorithm.ppo.file_io.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(OSError, "disk write failure"):
                replace_file(source, target, Path("fallback.pt"))
        sleep.assert_not_called()

    def test_both_destinations_denied_preserves_temporary_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            source = Path(directory) / "checkpoint.tmp"
            source.write_bytes(b"complete-and-recoverable")
            with (
                patch.object(Path, "replace", side_effect=PermissionError("directory denied")),
                patch("algorithm.ppo.file_io.time.sleep"),
            ):
                with self.assertRaises(PermissionError):
                    replace_file(source, source.with_suffix(".pt"), Path(directory) / "fallback.pt")
            self.assertEqual(source.read_bytes(), b"complete-and-recoverable")

    @unittest.skipUnless(os.name == "nt", "Windows file sharing semantics")
    def test_real_windows_handle_blocks_replace_then_recovers(self):
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            source, target = Path(directory) / "new.tmp", Path(directory) / "old.pt"
            source.write_bytes(b"new-data")
            target.write_bytes(b"old-data")
            # FILE_SHARE_READ | FILE_SHARE_WRITE intentionally omits SHARE_DELETE.
            handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            closed = False

            def release_handle(delay):
                nonlocal closed
                self.assertTrue(kernel.CloseHandle(handle))
                closed = True

            try:
                with self.assertRaises(PermissionError):
                    source.replace(target)
                with patch("algorithm.ppo.file_io.time.sleep", side_effect=release_handle) as sleep:
                    self.assertEqual(replace_file(source, target), target)
                self.assertEqual(sleep.call_count, 1)
                self.assertEqual(target.read_bytes(), b"new-data")
            finally:
                if not closed:
                    kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
