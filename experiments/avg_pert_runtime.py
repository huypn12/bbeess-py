import numpy as np
from datetime import datetime

checkpoints = [
    "2021-04-06 13:07:26.911444",
    "2021-04-06 13:53:47.035361",
    "2021-04-06 13:53:47.035658",
    "2021-04-06 14:42:56.003897",
    "2021-04-06 14:42:56.004193",
    "2021-04-06 15:31:07.408461",
    "2021-04-06 15:31:07.408737",
    "2021-04-06 16:16:49.425176",
    "2021-04-06 16:16:49.425469",
    "2021-04-06 17:03:29.076421",
    "2021-04-06 17:03:29.076704",
    "2021-04-06 17:38:35.567870",
    "2021-04-06 17:38:35.568158",
    "2021-04-06 18:13:15.888731",
    "2021-04-06 18:13:15.889017",
    "2021-04-06 18:52:12.019422",
    "2021-04-06 18:52:12.019724",
    "2021-04-06 19:29:12.070234",
    "2021-04-06 19:29:12.070475",
    "2021-04-06 20:05:50.427617",
    "2021-04-06 20:05:50.427842",
    "2021-04-06 20:44:21.038914",
    "2021-04-06 20:44:21.039261",
    "2021-04-06 21:23:00.759859",
    "2021-04-06 21:23:00.760225",
    "2021-04-06 21:57:48.631299",
    "2021-04-06 21:57:48.631667",
    "2021-04-06 22:28:58.044379",
    "2021-04-06 22:28:58.044745",
    "2021-04-06 23:00:27.478177",
    "2021-04-06 23:00:27.478404",
    "2021-04-06 23:31:02.411880",
    "2021-04-06 23:31:02.412168",
    "2021-04-07 00:04:13.072447",
    "2021-04-07 00:04:13.072740",
    "2021-04-07 00:36:55.431096",
    "2021-04-07 00:36:55.431318",
    "2021-04-07 01:08:23.926929",
]


def avg_iter_time(checkpoints):
    iter_times = []
    i = 0
    while i < len(checkpoints):
        start_time_str = checkpoints[i]
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
        end_time_str = checkpoints[i + 1]
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.%f")
        delta = end_time - start_time
        iter_times.append(delta.total_seconds())
        i += 2
    print(sum(iter_times) * 1.0 / (60 * len(iter_times)))


def main():
    avg_iter_time(checkpoints)


if __name__ == "__main__":
    main()