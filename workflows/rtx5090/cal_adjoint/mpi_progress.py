"""Single aggregate tqdm bar across MPI ranks (pattern: worker ticks -> rank 0).

Usage:
    with mpi_tqdm(comm, total=len(files), unit="file") as tick:
        for item in files[rank::size]:
            do_work(item)
            tick()

Workers send a 1-int message to rank 0 after each unit of work. Rank 0 holds
the only visible bar; between its own ticks it drains pending worker messages
via Iprobe so the bar stays responsive. After rank 0 finishes its own loop,
the context manager blocks until all expected ticks have arrived.

Falls back to a plain local tqdm when comm is None or size == 1.
"""

from __future__ import annotations

from contextlib import contextmanager

from tqdm import tqdm

_TICK_TAG = 0xBAD


@contextmanager
def mpi_tqdm(comm, total: int, **tqdm_kwargs):
    if comm is None:
        rank, size = 0, 1
    else:
        rank = comm.Get_rank()
        size = comm.Get_size()

    if size == 1:
        bar = tqdm(total=total, **tqdm_kwargs)
        try:
            yield lambda: bar.update(1)
        finally:
            bar.close()
        return

    from mpi4py import MPI

    if rank == 0:
        bar = tqdm(total=total, **tqdm_kwargs)
        status = MPI.Status()

        def tick() -> None:
            bar.update(1)
            while comm.Iprobe(source=MPI.ANY_SOURCE, tag=_TICK_TAG, status=status):
                comm.recv(source=status.Get_source(), tag=_TICK_TAG)
                bar.update(1)

        try:
            yield tick
            while bar.n < total:
                comm.recv(source=MPI.ANY_SOURCE, tag=_TICK_TAG)
                bar.update(1)
        finally:
            bar.close()
    else:
        def tick() -> None:
            comm.send(1, dest=0, tag=_TICK_TAG)

        yield tick
