"""The worker pool: inline and thread backends behave the same."""

from __future__ import annotations

import threading
import unittest

from parsec_python.acceleration.Parallel import WorkerError, make_pool, resolve_workers


class _Worker:
    def __init__(self, tasks, index):
        self.tasks = tasks
        self.index = index
        self.memory = {task: task * 10 for task in tasks}  # state kept between calls
        self.thread = threading.get_ident()

    def scaled(self, factor):
        for task in self.tasks:
            yield task, self.memory[task] * factor

    def total(self):
        yield None, sum(self.memory.values())

    def owner(self):
        for task in self.tasks:
            yield task, (self.index, self.thread)

    def fail(self):
        raise ValueError("bad task")
        yield  # pragma: no cover


class WorkerPoolTests(unittest.TestCase):
    def test_every_task_is_computed_once_for_any_worker_count(self) -> None:
        for workers in (1, 2, 3, 7):
            with self.subTest(workers=workers), make_pool(range(7), _Worker, workers=workers) as pool:
                self.assertEqual(pool.by_task("scaled", 2), {t: 20 * t for t in range(7)})
                self.assertEqual(sum(part for _, part in pool.results("total")), 210)

    def test_state_stays_with_the_worker_between_calls(self) -> None:
        with make_pool(range(6), _Worker, workers=3) as pool:
            first = pool.by_task("owner")
            second = pool.by_task("owner")
            self.assertEqual(first, second)
            self.assertEqual(len({index for index, _ in first.values()}), 3)

    def test_tasks_are_dealt_round_robin(self) -> None:
        with make_pool(range(6), _Worker, workers=3) as pool:
            owners = {task: index for task, (index, _) in pool.by_task("owner").items()}
        self.assertEqual(owners, {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2})

    def test_threads_are_distinct_and_the_serial_path_uses_the_caller(self) -> None:
        with make_pool(range(4), _Worker, workers=2) as pool:
            self.assertEqual(pool.kind, "thread")
            threads = {thread for _, thread in pool.by_task("owner").values()}
            self.assertEqual(len(threads), 2)
            self.assertNotIn(threading.get_ident(), threads)
        with make_pool(range(4), _Worker, workers=1) as pool:
            self.assertEqual(pool.kind, "inline")
            self.assertEqual({t for _, t in pool.by_task("owner").values()}, {threading.get_ident()})

    def test_results_stream_as_they_are_produced(self) -> None:
        with make_pool(range(4), _Worker, workers=2) as pool:
            stream = pool.stream("scaled", 1)
            self.assertEqual(next(stream)[1] % 10, 0)
            self.assertEqual(len(list(stream)), 3)

    def test_worker_errors_are_raised_with_their_traceback(self) -> None:
        with make_pool(range(4), _Worker, workers=2) as pool:
            with self.assertRaises(WorkerError) as caught:
                pool.results("fail")
            self.assertIn("ValueError: bad task", str(caught.exception))

    def test_a_failing_factory_is_reported_on_the_first_call(self) -> None:
        def factory(tasks, index):
            raise RuntimeError("cannot build")

        pool = make_pool(range(4), factory, workers=2)
        with self.assertRaises(WorkerError) as caught:
            pool.results("scaled", 1)
        self.assertIn("cannot build", str(caught.exception))

    def test_worker_count_resolution(self) -> None:
        self.assertEqual(resolve_workers(5, 3), 3)
        self.assertEqual(resolve_workers(1, 3), 1)
        self.assertEqual(resolve_workers("auto", 1), 1)
        self.assertEqual(resolve_workers("auto", 8, problem_size=100), 1)


if __name__ == "__main__":
    unittest.main()
