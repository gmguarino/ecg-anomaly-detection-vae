import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        raise TimeoutError
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


def my_func():
    # Add a timeout block.
    try:
        with timeout(1):
            print('entering block')
            import time
            time.sleep(10)
            print('This should never get printed because the line before timed out')
    except TimeoutError:
        print("Timeout")

if __name__ == '__main__':
    my_func()
