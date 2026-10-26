"""Observe controller acknowledgements without changing acceptance deadlines."""
import time


def track_ack(future, joints, sent, deadline, records):
    row = dict(joints=list(joints), sent_monotonic_s=sent, deadline_monotonic_s=deadline)
    records.append(row)
    def received(reply):
        now = time.monotonic()
        row.update(received_monotonic_s=now, latency_ms=(now-sent)*1000, on_time=now<deadline)
        try:
            row['accepted'] = bool(reply.result().accepted)
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
    future.add_done_callback(received)
    return future
