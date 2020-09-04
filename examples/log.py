from time import sleep
from random import uniform
from pytools.log import (LogManager, add_general_quantities,
        add_simulation_quantities, add_run_info, IntervalTimer,
        LogQuantity, set_dt)

from warnings import warn


class Fifteen(LogQuantity):
    def __call__(self):
        return 15


def main():
    logmgr = LogManager("mylog.dat", "w")  # , comm=...

    # set a run property
    logmgr.set_constant("myconst", uniform(0, 1))

    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)

    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    logmgr.add_quantity(Fifteen("fifteen"))
    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    for istep in range(200):
        logmgr.tick_before()

        dt = uniform(0.01, 0.1)
        set_dt(logmgr, dt)
        sleep(dt)

        # Illustrate custom timers
        if istep % 10 == 0:
            with vis_timer.start_sub_timer():
                sleep(0.05)

        # Illustrate warnings capture
        if uniform(0, 1) < 0.05:
            warn("Oof. Something went awry.")

        logmgr.tick_after()

    logmgr.close()


if __name__ == "__main__":
    main()
