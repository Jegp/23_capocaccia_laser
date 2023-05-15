##
# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <jens@jepedersen.dk> wrote this file.  As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer in return.    Jens E. Pedersen
# ----------------------------------------------------------------------------
##
 
import math
import serial
import time


class _Getch:
    """Gets a single character from standard input.  Does not echo to the
    screen.
        Thanks to https://stackoverflow.com/a/510364"""

    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt

        return msvcrt.getch()


getch = _Getch()


class Laser:
    def __init__(
        self, baud=8000000, device="/dev/ttyUSB0", timeout=1, write_timeout=0.1
    ):
        # picocom --b 8000000 --f h --imap lfcrlf /dev/ttyUSB0
        self.baud = baud
        self.connection = serial.Serial(device, baud)
        print("Connected to laser")

    def __enter__(self):
        # self.connection.open()
        return self

    def __exit__(self, a, b, c):
        self.connection.close()

    def send(self, command):
        self.connection.write(bytes(command + "\r\n", "ascii"))
        self.connection.flush()

    def move(self, x, y):
        self.send(f"!L{x:03X}{y:03X}")

    def on(self):
        self.send("!L=+")

    def blink(self, period):
        self.send(f"!L={period}")

    def off(self):
        self.send("!L=-")

    def read(self):
        return self.connection.read_until("\n\r")


def parse_char(laser, ch, state=None):
    moves = {"w": (-100, 0), "s": (100, 0), "a": (0, -100), "d": (0, 100)}

    if state is None:
        state = (4095, 4095)

    if ch in moves.keys():
        state = (state[0] + moves[ch][0], state[1] + moves[ch][1])
        state = (min(4095, max(0, state[0])), min(4095, max(0, state[1])))
        laser.move(*state)
    elif ch == "q":
        laser.off()
        return None
    else:
        print("Unknown input", bytes(ch, "ascii"))

    return state


if __name__ == "__main__":
    state = None
    with Laser() as l:
        l.on()
        while True:
            ch = getch()
            state = parse_char(l, ch, state)
            if state == None:
                break
