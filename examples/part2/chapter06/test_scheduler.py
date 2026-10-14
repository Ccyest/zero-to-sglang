"""Check scheduling order, output isolation, and capacity release; no third-party packages."""

import unittest

from .scheduler import Scheduler


class SchedulerTests(unittest.TestCase):
    def test_late_request_uses_freed_slot(self):
        scheduler = Scheduler()
        a = scheduler.submit("A", [1, 2], 2)
        b = scheduler.submit("B", [3, 4, 5], 5)
        trace = [scheduler.step()]
        c = scheduler.submit("C", [6], 2)
        while scheduler.has_work:
            trace.append(scheduler.step())
        self.assertEqual(trace, [
            ("prefill", ["A", "B"], 12), ("decode", ["A", "B"], 8),
            ("prefill", ["C"], 11), ("decode", ["B", "C"], 8),
            ("decode", ["B"], 8), ("decode", ["B"], 0),
        ])
        for request in (a, b, c):
            expected = [(request.prompt[-1] + i + 1) % 100
                        for i in range(request.max_new_tokens)]
            self.assertEqual(request.output, expected)
            self.assertEqual(request.state, "finished")
        self.assertEqual(scheduler.reserved, 0)
        self.assertIsNone(scheduler.step())

    def test_cancellation_in_both_queues(self):
        scheduler = Scheduler(max_running=1)
        a = scheduler.submit("A", [1], 3)
        b = scheduler.submit("B", [2], 3)
        scheduler.step()
        self.assertTrue(scheduler.cancel("B"))
        self.assertTrue(scheduler.cancel("A"))
        self.assertFalse(scheduler.cancel("A"))
        self.assertFalse(scheduler.cancel("unknown"))
        self.assertEqual((a.state, b.state), ("cancelled", "cancelled"))
        self.assertFalse(scheduler.has_work)
        self.assertEqual(scheduler.reserved, 0)

    def test_eos_and_zero_length(self):
        scheduler = Scheduler()
        a = scheduler.submit("A", [98], 5, eos_id=99)
        zero = scheduler.submit("zero", [1], 0)
        scheduler.step()
        self.assertEqual(a.output, [99])
        self.assertEqual(a.finish_reason, "eos")
        self.assertEqual(zero.output, [])
        self.assertEqual(zero.finish_reason, "length")
        self.assertFalse(scheduler.has_work)

    def test_capacity_and_prefill_budget(self):
        for options, expected in (({"token_capacity": 6}, [
            ("prefill", ["A"]), ("decode", ["A"]), ("prefill", ["B"]),
        ]), ({"prefill_budget": 2}, [
            ("prefill", ["A"]), ("prefill", ["B"]), ("decode", ["A", "B"]),
        ])):
            with self.subTest(options=options):
                scheduler = Scheduler(**options)
                scheduler.submit("A", [1, 2], 2)
                scheduler.submit("B", [3], 2)
                self.assertEqual([scheduler.step()[:2] for _ in range(3)], expected)
                while scheduler.has_work:
                    scheduler.step()
                self.assertEqual(scheduler.reserved, 0)

    def test_fifo_does_not_skip_blocked_head(self):
        scheduler = Scheduler(token_capacity=8)
        scheduler.submit("A", [1], 3)  # Reserves 4 tokens.
        scheduler.step()
        scheduler.submit("B", [2], 4)  # Needs 5 tokens; cannot fit yet.
        scheduler.submit("C", [3], 1)  # Needs only 2 tokens, but cannot bypass B.
        self.assertEqual(scheduler.step()[1], ["A"])
        scheduler.step()
        self.assertEqual(scheduler.step()[1], ["B", "C"])

    def test_invalid_requests_never_enter_queue(self):
        scheduler = Scheduler()
        for prompt, length in (([], 1), ([1], -1), ([1] * 9, 1), ([1], 16)):
            with self.assertRaises(ValueError):
                scheduler.submit("bad", prompt, length)
        self.assertFalse(scheduler.has_work)
        scheduler.submit("A", [1], 1)
        with self.assertRaises(ValueError):
            scheduler.submit("A", [1], 1)

    def test_backend_failure_releases_batch(self):
        def fail(batch, phase):
            raise RuntimeError("backend failed")

        for engine, error in ((fail, RuntimeError), (lambda batch, phase: [], ValueError)):
            scheduler = Scheduler(engine=engine)
            request = scheduler.submit("A", [1], 2)
            with self.assertRaises(error):
                scheduler.step()
            self.assertEqual(request.finish_reason, "error")
            self.assertEqual(scheduler.reserved, 0)
            self.assertFalse(scheduler.has_work)


if __name__ == "__main__":
    unittest.main()
