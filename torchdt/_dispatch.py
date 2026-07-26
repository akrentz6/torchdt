from threading import local


class _LocalDispatch(local):
    """A reentrant dispatch slot isolated for each execution thread."""

    current = None

    def get(self):
        return self.current

    def set(self, value):
        previous = self.current
        self.current = value
        return previous

    def reset(self, previous):
        self.current = previous


# Torch operations are synchronous within a Python thread, so a thread-local
# stack slot provides the required nesting/concurrency isolation without the
# substantially higher ContextVar cost on every eager operation.
current_dispatch = _LocalDispatch()
