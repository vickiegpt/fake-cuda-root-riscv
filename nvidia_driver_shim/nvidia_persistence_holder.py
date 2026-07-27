#!/usr/bin/env python3

import argparse
import os
import signal
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep the Lanxin NVIDIA RM/GSP instance initialized."
    )
    parser.add_argument(
        "--device",
        default="/dev/nvidia0",
        help="NVIDIA GPU device node to hold open",
    )
    parser.add_argument(
        "--control",
        default="/dev/nvidiactl",
        help="NVIDIA control device node to hold open",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    descriptors = []
    for path in (args.control, args.device):
        descriptors.append(os.open(path, os.O_RDWR | os.O_CLOEXEC))

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"holding {args.control} and {args.device}; pid={os.getpid()}",
        flush=True,
    )
    while not stopping:
        signal.pause()

    for descriptor in reversed(descriptors):
        os.close(descriptor)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as error:
        print(f"failed to open NVIDIA device: {error}", file=sys.stderr)
        sys.exit(1)
