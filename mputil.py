# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import time


class Timer(object):
    """ Convenience class for timing code w/ a context timer.

    Note that timer prints wall time, but also keeps track of
    cpu time internally (you can access via timer.interval_cpu

    Not super accurate, but super convenient!

    with Timer('Name of timer'):
        do_slow_stuff(...)
        do_some_other_stuff(...)

    # Once code reaches here, timing and name info will be printed to stdout.

    """

    def __init__(self, name=__name__, log_output=True):
        self.name = name
        self.log_output = log_output

    def __enter__(self):
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        if self.log_output:
            print(f"/ / / /   [{self.name} ...]")
        return self

    def __exit__(self, *args):
        self.end_wall = time.perf_counter()
        self.end_cpu = time.process_time()
        self.interval_wall = self.end_wall - self.start_wall
        self.interval_cpu = self.end_cpu - self.start_cpu
        if self.log_output:
            print(
                f"\\ \\ \\ \\  {self.interval_wall:.03f} s wall  ({self.interval_cpu:.03f} s cpu) [... {self.name}]\n"
            )
