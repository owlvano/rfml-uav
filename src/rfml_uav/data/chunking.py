import numpy as np


def flag_duplicates(starts, chunk_size: int, max_overlap: float = 0.5) -> np.ndarray:
    """Flag windows that overlap the last kept window of the same recording by > max_overlap * chunk_size.

    `starts` must be ascending (chunking stores them sorted). Returns a bool mask, True = duplicate.
    """
    starts = np.asarray(starts)
    is_dup = np.zeros(len(starts), dtype=bool)
    last_kept = None
    for i, s in enumerate(starts):
        if last_kept is not None and chunk_size - (s - last_kept) > max_overlap * chunk_size:
            is_dup[i] = True
        else:
            last_kept = s
    return is_dup


if __name__ == "__main__":
    # L/H offset duplicate: second window overlaps first by 3900/4000
    assert flag_duplicates([1000, 1100], 4000).tolist() == [False, True]
    # exactly 50% overlap is kept
    assert flag_duplicates([0, 2000], 4000).tolist() == [False, False]
    # contiguous background windows
    assert not flag_duplicates(np.arange(0, 40000, 4000), 4000).any()
    # compared against last *kept* window: 1000 dup of 0; 2500 overlaps 0 by 1500 -> kept; 3000 dup of 2500
    assert flag_duplicates([0, 1000, 2500, 3000], 4000).tolist() == [False, True, False, True]
    print("chunking self-check OK")
