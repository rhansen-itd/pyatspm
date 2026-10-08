# Shell tests for retrieval's eos-time.jsonl fetch: merging into the local
# copy, and which devices are asked for a send log.  The SSH session is a
# stub; nothing touches the network.

import io
import json

from atspm.data.retrieval import SEND_LOG_NAME, RetrievalEngine, merge_send_log


def _rec(start):
    return json.dumps({"mode": "drift-check", "start": start, "pulses": []})


A, B, C = _rec("2026-10-08T01:37:01"), _rec("2026-10-08T02:37:01"), _rec("2026-10-08T03:37:01")


class _File(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Sftp:
    def __init__(self, files):
        self.files, self.opened = files, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def open(self, path, mode="r"):
        self.opened.append(path)
        if path not in self.files:
            raise IOError(2, "No such file")
        return _File(self.files[path].encode())


class _Ssh:
    def __init__(self, files):
        self.sftp = _Sftp(files)

    def open_sftp(self):
        return self.sftp


DEFAULT = "eos_set_time_standalone/eos-time.jsonl"


class TestMergeSendLog:

    def test_creates_the_local_copy(self, tmp_path):
        path = tmp_path / SEND_LOG_NAME
        assert merge_send_log(path, [A, B, ""]) == 2
        assert path.read_text().splitlines() == [A, B]

    def test_appends_only_new_records_and_keeps_local_history(self, tmp_path):
        # The head unit's log was rotated: A is gone remotely but stays here.
        path = tmp_path / SEND_LOG_NAME
        path.write_text(A + "\n" + B + "\n")
        assert merge_send_log(path, [B, C]) == 1
        assert path.read_text().splitlines() == [A, B, C]

    def test_nothing_new_leaves_the_file_alone(self, tmp_path):
        path = tmp_path / SEND_LOG_NAME
        path.write_text(A + "\n")
        before = path.stat().st_mtime_ns
        assert merge_send_log(path, [A]) == 0
        assert path.stat().st_mtime_ns == before


class TestPullSendLog:

    def _engine(self, tmp_path):
        return RetrievalEngine(tmp_path, {}, [])

    def test_secondary_defaults_to_the_standalone_log(self, tmp_path):
        ssh = _Ssh({DEFAULT: A + "\n" + B + "\n"})
        n = self._engine(tmp_path)._pull_send_log(ssh, {"role": "secondary"})
        assert n == 2
        assert (tmp_path / SEND_LOG_NAME).read_text().splitlines() == [A, B]

    def test_controller_has_no_default(self, tmp_path):
        ssh = _Ssh({DEFAULT: A})
        assert self._engine(tmp_path)._pull_send_log(ssh, {"role": "controller"}) == 0
        assert ssh.sftp.opened == []

    def test_null_turns_the_fetch_off(self, tmp_path):
        ssh = _Ssh({DEFAULT: A})
        device = {"role": "secondary", "send_log": None}
        assert self._engine(tmp_path)._pull_send_log(ssh, device) == 0
        assert ssh.sftp.opened == []

    def test_explicit_path_overrides_the_default(self, tmp_path):
        ssh = _Ssh({"/opt/eos/eos-time.jsonl": C})
        device = {"role": "controller", "send_log": "/opt/eos/eos-time.jsonl"}
        assert self._engine(tmp_path)._pull_send_log(ssh, device) == 1

    def test_missing_remote_log_is_not_an_error(self, tmp_path):
        assert self._engine(tmp_path)._pull_send_log(_Ssh({}), {"role": "secondary"}) == 0
        assert not (tmp_path / SEND_LOG_NAME).exists()
