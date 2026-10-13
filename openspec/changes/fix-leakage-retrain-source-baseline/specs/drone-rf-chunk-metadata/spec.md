# Spec Delta

## Purpose

Keeps the provenance of every DroneRF chunk (source recording, position in the recording, and duplicate status) from chunking through feature extraction, so that downstream splits, streams and dedup can operate per recording and in temporal order.

## ADDED Requirements

### Requirement: Chunk start positions are persisted
The chunking stage SHALL store, for every chunk of every `(bui, n)` recording, the start sample index of the window in the raw recording, alongside the chunk data. Chunks SHALL be stored in ascending start order. The existing chunk array SHALL remain unchanged in content and key so existing readers keep working.

#### Scenario: Start index stored per chunk
- **WHEN** a recording `{bui}_{n}` is chunked into K windows
- **THEN** its chunked file contains K chunk arrays and K start indices, with `chunk[i] == raw[start[i] : start[i] + chunk_size]`

#### Scenario: Legacy readers unaffected
- **WHEN** a notebook loads `chunked/{bui}L_{n}.npz` and reads key `arr_0`
- **THEN** it receives the same chunk array as before this change

### Requirement: Features carry per-row provenance
The WPD feature stage SHALL emit, in one file, the feature matrix together with per-row `bui`, `n`, `chunk_idx` and `start`, in recording and temporal order.

#### Scenario: Provenance lookup
- **WHEN** a consumer loads the WPD feature file
- **THEN** every feature row can be traced to its `(bui, n, chunk_idx, start)`, and rows from the same recording appear in ascending `start`

### Requirement: Near-duplicate windows are flagged
Each feature row SHALL carry an `is_dup` flag that is true when its window overlaps a previously kept window of the same recording by more than 50% of the chunk length. Flagged rows SHALL be kept in the file, not deleted, so experiments can include or exclude them.

#### Scenario: L/H offset duplicate
- **WHEN** a packet produces two windows in one recording with starts 1000 and 1100 (chunk length 4000)
- **THEN** the first row has `is_dup = false` and the second has `is_dup = true`

#### Scenario: Non-overlapping windows
- **WHEN** two windows of the same recording overlap by at most 50%
- **THEN** neither is flagged because of the other

#### Scenario: Background recordings
- **WHEN** a background (`00000`) recording is chunked into contiguous non-overlapping windows
- **THEN** no row of that recording is flagged
