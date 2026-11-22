"""Loop detector for note 020's loop rate ("generations stopped early for a repeated tail block")."""


def tail_loop(tokens, min_block=8, max_block=256, reps=3):
    """True if the sequence ends with >= reps consecutive copies of a block of min_block..max_block tokens."""
    n = len(tokens)
    for b in range(min_block, min(max_block, n // reps) + 1):
        blk = tokens[n - b:]
        if all(tokens[n - (r + 1) * b:n - r * b] == blk for r in range(1, reps)):
            return True
    return False


def first_loop_stop(tokens, check_every=16, **kw):
    """Index where an online detector would stop generation (None if never)."""
    for t in range(check_every, len(tokens) + 1, check_every):
        if tail_loop(tokens[:t], **kw):
            return t
    return None
