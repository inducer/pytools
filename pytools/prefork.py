"""OpenMPI, once initialized, prohibits forking. This helper module
allows the forking of *one* helper child process before OpenMPI
initialization that can do the forking for the fork-challenged
parent process.

Since none of this is MPI-specific, it got parked in :mod:`pytools`.

.. autoexception:: ExecError
    :show-inheritance:

.. autoclass:: Forker
.. autoclass:: DirectForker
.. autoclass:: IndirectForker

.. autofunction:: enable_prefork
.. autofunction:: call
.. autofunction:: call_async
.. autofunction:: call_capture_output
.. autofunction:: wait
.. autofunction:: waitall
"""
from __future__ import annotations

import socket
from abc import ABC, abstractmethod
from subprocess import Popen
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


class ExecError(OSError):
    pass


class Forker(ABC):
    @abstractmethod
    def call(self, cmdline: Sequence[str], cwd: str | None = None) -> int:
        pass

    @abstractmethod
    def call_async(self, cmdline: Sequence[str], cwd: str | None = None) -> int:
        pass

    @abstractmethod
    def call_capture_output(self,
                            cmdline: Sequence[str],
                            cwd: str | None = None,
                            error_on_nonzero: bool = True) -> tuple[int, bytes, bytes]:
        pass

    @abstractmethod
    def wait(self, aid: int) -> int:
        pass

    @abstractmethod
    def waitall(self) -> dict[int, int]:
        pass


class DirectForker(Forker):
    def __init__(self) -> None:
        self.apids: dict[int, Popen[bytes]] = {}
        self.count: int = 0

    def call(self, cmdline: Sequence[str], cwd: str | None = None) -> int:
        from subprocess import call as spcall

        try:
            return spcall(cmdline, cwd=cwd)
        except OSError as e:
            raise ExecError(
                    "error invoking '{}': {}".format(" ".join(cmdline), e)) from e

    def call_async(self, cmdline: Sequence[str], cwd: str | None = None) -> int:
        try:
            self.count += 1

            proc = Popen(cmdline, cwd=cwd)
            self.apids[self.count] = proc

            return self.count
        except OSError as e:
            raise ExecError(
                "error invoking '{}': {}".format(" ".join(cmdline), e)) from e

    def call_capture_output(self,
                            cmdline: Sequence[str],
                            cwd: str | None = None,
                            error_on_nonzero: bool = True) -> tuple[int, bytes, bytes]:
        from subprocess import PIPE, Popen

        try:
            popen = Popen(cmdline, cwd=cwd, stdin=PIPE, stdout=PIPE,
                          stderr=PIPE)
            stdout_data, stderr_data = popen.communicate()

            if error_on_nonzero and popen.returncode:
                raise ExecError("status {} invoking '{}': {}".format(
                    popen.returncode,
                    " ".join(cmdline),
                    stderr_data.decode("utf-8", errors="replace")))

            return popen.returncode, stdout_data, stderr_data
        except OSError as e:
            raise ExecError(
                    "error invoking '{}': {}".format(" ".join(cmdline), e)) from e

    def wait(self, aid: int) -> int:
        proc = self.apids.pop(aid)
        retc = proc.wait()

        return retc

    def waitall(self) -> dict[int, int]:
        rets = {}

        for aid in self.apids:
            rets[aid] = self.wait(aid)

        return rets


def _send_packet(sock: socket.socket, data: object) -> None:
    from pickle import dumps
    from struct import pack

    packet = dumps(data)

    sock.sendall(pack("I", len(packet)))
    sock.sendall(packet)


def _recv_packet(sock: socket.socket,
                 who: str = "Process",
                 partner: str = "other end") -> tuple[object, ...]:
    from struct import calcsize, unpack
    size_bytes_size = calcsize("I")
    size_bytes = sock.recv(size_bytes_size)

    if len(size_bytes) < size_bytes_size:
        raise SystemExit

    size, = unpack("I", size_bytes)

    packet = b""
    while len(packet) < size:
        packet += sock.recv(size)

    from pickle import loads

    result = loads(packet)
    assert isinstance(result, tuple)

    return result


def _fork_server(sock: socket.socket) -> None:
    # Ignore keyboard interrupts, we'll get notified by the parent.
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Construct a local DirectForker to do the dirty work
    df = DirectForker()

    funcs = {
        "call": df.call,
        "call_async": df.call_async,
        "call_capture_output": df.call_capture_output,
        "wait": df.wait,
        "waitall": df.waitall
    }

    try:
        while True:
            func_name, args, kwargs = _recv_packet(
                sock, who="Prefork server", partner="parent"
            )
            assert isinstance(func_name, str)

            if func_name == "quit":
                df.waitall()
                _send_packet(sock, ("ok", None))
                break
            try:
                result = funcs[func_name](*args, **kwargs)  # type: ignore[operator]
            # FIXME: Is catching all exceptions the right course of action?
            except Exception as e:  # pylint:disable=broad-except
                _send_packet(sock, ("exception", e))
            else:
                _send_packet(sock, ("ok", result))
    finally:
        sock.close()

    import os
    os._exit(0)


class IndirectForker(Forker):
    def __init__(self, server_pid: int, sock: socket.socket) -> None:
        self.server_pid = server_pid
        self.socket = sock

        import atexit
        atexit.register(self._quit)

    def _remote_invoke(self, name: str, *args: Any, **kwargs: Any) -> object:
        _send_packet(self.socket, (name, args, kwargs))
        status, result = _recv_packet(
            self.socket, who="Prefork client", partner="prefork server"
        )

        if status == "exception":
            assert isinstance(result, Exception)
            raise result

        assert status == "ok"
        return result

    def _quit(self) -> None:
        self._remote_invoke("quit")

        from os import waitpid
        waitpid(self.server_pid, 0)

    def call(self, cmdline: Sequence[str], cwd: str | None = None) -> int:
        result = self._remote_invoke("call", cmdline, cwd)

        assert isinstance(result, int)
        return result

    def call_async(self, cmdline: Sequence[str], cwd: str | None = None) -> int:
        result = self._remote_invoke("call_async", cmdline, cwd)

        assert isinstance(result, int)
        return result

    def call_capture_output(self,
                            cmdline: Sequence[str],
                            cwd: str | None = None,
                            error_on_nonzero: bool = True,
                            ) -> tuple[int, bytes, bytes]:
        return self._remote_invoke("call_capture_output", cmdline, cwd,  # type: ignore[return-value]
                                   error_on_nonzero)

    def wait(self, aid: int) -> int:
        result = self._remote_invoke("wait", aid)

        assert isinstance(result, int)
        return result

    def waitall(self) -> dict[int, int]:
        result = self._remote_invoke("waitall")

        assert isinstance(result, dict)
        return result


forker: Forker = DirectForker()


def enable_prefork() -> None:
    global forker

    if isinstance(forker, IndirectForker):
        return

    s_parent, s_child = socket.socketpair()

    from os import fork
    fork_res = fork()

    # Child
    if fork_res == 0:
        s_parent.close()
        _fork_server(s_child)
    # Parent
    else:
        s_child.close()
        forker = IndirectForker(fork_res, s_parent)


def call(cmdline: Sequence[str], cwd: str | None = None) -> int:
    return forker.call(cmdline, cwd)


def call_async(cmdline: Sequence[str], cwd: str | None = None) -> int:
    return forker.call_async(cmdline, cwd)


def call_capture_output(cmdline: Sequence[str],
                        cwd: str | None = None,
                        error_on_nonzero: bool = True) -> tuple[int, bytes, bytes]:
    return forker.call_capture_output(cmdline, cwd, error_on_nonzero)


def wait(aid: int) -> int:
    return forker.wait(aid)


def waitall() -> dict[int, int]:
    return forker.waitall()
