"""A killed run's lock must not block every later run; a live one still must."""

import os
import socket
import subprocess

import pytest

from eval_tool.infer_cache import InferCacheError, InferenceLock


def _dead_pid():
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_stale_lock_from_this_host_is_taken_over(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text(f"pid={_dead_pid()}\nhost={socket.gethostname()}\n", encoding="utf-8")
    with InferenceLock(lock):
        assert f"pid={os.getpid()}" in lock.read_text(encoding="utf-8")
    assert not lock.exists()


@pytest.mark.parametrize("content", [
    f"pid={os.getpid()}\nhost={socket.gethostname()}\n",     # alive, here
    "pid=12345\nhost=some-other-node\n",                      # another machine on the shared disk
    "pid=12345\n",                                            # old format: cannot tell
])
def test_live_or_unverifiable_lock_still_blocks(tmp_path, content):
    lock = tmp_path / "x.lock"
    lock.write_text(content, encoding="utf-8")
    with pytest.raises(InferCacheError, match="pid=12345|pid=" + str(os.getpid())):
        with InferenceLock(lock):
            pass
    assert lock.read_text(encoding="utf-8") == content
