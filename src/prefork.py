"""OpenMPI, once intialized, prohibits forking. This helper module
allows the forking of *one* helper child process before OpenMPI 
initializaton that can do the forking for the fork-challenged 
parent process.

Since none of this is MPI-specific, it got parked in pytools.
"""





class DirectForker:
    @staticmethod
    def call(cmdline, cwd=None):
        from subprocess import call
        return call(cmdline, cwd=cwd)

    @staticmethod
    def call_capture_stdout(cmdline, cwd=None):
        from subprocess import Popen, PIPE
        return Popen(cmdline, cwd=cwd, stdout=PIPE).communicate()[0]




def _send_packet(sock, data):
    from struct import pack
    from cPickle import dumps

    packet = dumps(data)
    
    sock.sendall(pack("I", len(packet)))
    sock.sendall(packet)

def _recv_packet(sock):
    from struct import calcsize, unpack
    size, = unpack("I", sock.recv(calcsize("I")))

    packet = ""
    while len(packet) < size:
        packet += sock.recv(size)

    from cPickle import loads
    return loads(packet)




def _fork_server(sock):
    import signal
    # ignore keyboard interrupts, we'll get notified by the parent.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    quitflag = [False]

    def quit():
        quitflag[0] = True

    funcs = {
    "quit": quit,
    "call": DirectForker.call,
    "call_capture_stdout": DirectForker.call_capture_stdout,
    }

    try:
        while not quitflag[0]:
            func_name, args, kwargs = _recv_packet(sock)

            try:
                result = funcs[func_name](*args, **kwargs)
            except Exception, e:
                _send_packet(sock, ("exception", e))
            else:
                _send_packet(sock, ("ok", result))
    finally:
        sock.close()

    import sys
    sys.exit(0)





class IndirectForker:
    def __init__(self, server_pid, sock):
        self.server_pid = server_pid
        self.socket = sock

    def _remote_invoke(self, name, *args, **kwargs):
        _send_packet(self.socket, (name, args, kwargs))
        status, result = _recv_packet(self.socket)
        
        if status == "exception":
            raise result
        elif status == "ok":
            return result

    def _quit(self):
        self._remote_invoke("quit")
        from os import waitpid
        waitpid(self.server_pid, 0)

    def call(self, cmdline, cwd=None):
        return self._remote_invoke("call", cmdline, cwd)

    def call_capture_stdout(self, cmdline, cwd=None):
        return self._remote_invoke("call_capture_stdout", cmdline, cwd)




def enable_prefork():
    if isinstance(forker[0], IndirectForker):
        return 

    from socket import socketpair
    s_parent, s_child = socketpair()

    from os import fork
    fork_res = fork()

    if fork_res == 0:
        # child
        s_parent.close()
        _fork_server(s_child)
    else:
        s_child.close()
        forker[0] = IndirectForker(fork_res, s_parent)

        import atexit
        atexit.register(forker[0]._quit)




forker = [DirectForker()]

def call(cmdline, cwd=None):
    return forker[0].call(cmdline, cwd)

def call_capture_stdout(cmdline, cwd=None):
    return forker[0].call_capture_stdout(cmdline, cwd)
