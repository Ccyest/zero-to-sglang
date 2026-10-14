"""Chapter 6: FIFO admission and prefill-first continuous batching; standard library only."""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    uid: str
    prompt: list[int]
    max_new_tokens: int
    eos_id: int | None = None
    output: list[int] = field(default_factory=list)
    state: str = "waiting"
    finish_reason: str | None = None

    @property
    def reservation(self):
        return len(self.prompt) + self.max_new_tokens


def toy_step(batch, phase):
    """Toy backend: return (last token + 1) % 100 per request; no real model or KV cache."""
    inputs = [r.prompt if phase == "prefill" else [r.output[-1]] for r in batch]
    return [(tokens[-1] + 1) % 100 for tokens in inputs]


class Scheduler:
    def __init__(self, engine=toy_step, max_running=2, token_capacity=16, prefill_budget=8):
        if min(max_running, token_capacity, prefill_budget) <= 0:
            raise ValueError("capacities must be positive")
        self.engine = engine
        self.max_running = max_running
        self.token_capacity = token_capacity
        self.prefill_budget = prefill_budget
        self.waiting = deque()
        self.running = []
        self.requests = {}

    @property
    def reserved(self):
        return sum(r.reservation for r in self.running)

    @property
    def has_work(self):
        return bool(self.waiting or self.running)

    def submit(self, uid, prompt, max_new_tokens, eos_id=None):
        if uid in self.requests:
            raise ValueError("request uid must be unique")
        if not prompt or max_new_tokens < 0:
            raise ValueError("nonempty prompt and nonnegative output length required")
        request = Request(uid, list(prompt), max_new_tokens, eos_id)
        if len(prompt) > self.prefill_budget or request.reservation > self.token_capacity:
            raise ValueError("request exceeds capacity; it would wait forever")
        self.requests[uid] = request
        self.waiting.append(request)
        if max_new_tokens == 0:
            self._finish(request, "length")
        return request

    def _finish(self, request, reason):
        if request.state == "waiting":
            self.waiting.remove(request)
        else:
            self.running.remove(request)
        request.state = "cancelled" if reason == "cancelled" else "finished"
        request.finish_reason = reason

    def cancel(self, uid):
        request = self.requests.get(uid)
        if request is None or request.state in {"finished", "cancelled"}:
            return False
        self._finish(request, "cancelled")
        return True

    def step(self):
        admitted = []
        budget = self.prefill_budget
        while self.waiting and len(self.running) < self.max_running:
            request = self.waiting[0]
            if len(request.prompt) > budget or self.reserved + request.reservation > self.token_capacity:
                break  # Strict FIFO: wait if the first request cannot fit this round.
            self.waiting.popleft()
            self.running.append(request)
            request.state = "running"
            budget -= len(request.prompt)
            admitted.append(request)

        phase = "prefill" if admitted else "decode"
        batch = admitted or list(self.running)
        if not batch:
            return None
        try:
            tokens = self.engine(batch, phase)  # One backend call per round.
            if len(tokens) != len(batch):
                raise ValueError("engine must return one token per request")
        except Exception:
            for request in batch:
                self._finish(request, "error")
            raise

        for request, token in zip(batch, tokens):
            request.output.append(token)
            if token == request.eos_id:
                self._finish(request, "eos")
            elif len(request.output) == request.max_new_tokens:
                self._finish(request, "length")
        return phase, [r.uid for r in batch], self.reserved


def main():
    scheduler = Scheduler()
    scheduler.submit("A", [1, 2], 2)
    scheduler.submit("B", [3, 4, 5], 5)
    tick = 0
    while scheduler.has_work:
        phase, batch, reserved = scheduler.step()
        print(f"tick={tick} {phase:7} batch={batch} reserved={reserved}")
        if tick == 0:
            scheduler.submit("C", [6], 2)  # C arrives after A and B have started.
        tick += 1
    for request in scheduler.requests.values():
        print(f"{request.uid}: {request.output} ({request.finish_reason})")


if __name__ == "__main__":
    main()
