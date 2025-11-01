# Checkpoint v2 CLI Usage Guide

This guide demonstrates how to use the checkpoint v2 CLI commands for debugging and time-travel.

## Prerequisites

Run a flow with checkpoint v2 enabled to create a `checkpoints.db` database. See `checkpoint_v2_basic_usage.py` for an example.

## CLI Commands

### List All Runs

View all checkpoint runs in the database:

```bash
python -m pydantic_flow debug list-runs checkpoints.db

# Limit results
python -m pydantic_flow debug list-runs checkpoints.db --limit 10
```

Output shows:
- Run ID
- Flow name
- Status
- Start time
- Duration
- Number of waves

### Show Run Timeline

View the execution timeline for a specific run:

```bash
python -m pydantic_flow debug timeline <run_id>

# Example
python -m pydantic_flow debug timeline UYe_4hlpdJKt...
```

Output shows each wave with:
- Wave number
- Timestamp
- Node IDs executed
- Execution duration

### Show Run Details

Get detailed information about a run:

```bash
python -m pydantic_flow debug details <run_id>
```

Output includes:
- Run metadata
- Full execution timeline
- State snapshots summary

### Replay from Checkpoint

Replay the recorded execution from a specific wave:

```bash
python -m pydantic_flow debug replay <run_id> --wave 1

# Default wave is 1
python -m pydantic_flow debug replay <run_id>
```

Shows:
- Node execution order
- Events captured (tokens, tools)
- Duration per node
- Cache hits

### Rewind to Previous Wave

Time-travel backward to reconstruct state at an earlier point:

```bash
python -m pydantic_flow debug rewind <run_id> --to-wave 5
```

This reconstructs the state as it was at wave 5, useful for:
- Understanding state evolution
- Debugging state-dependent issues
- Preparing to fork execution

### Fork Execution

Create a branching execution from a checkpoint:

```bash
python -m pydantic_flow debug fork <run_id> --from-wave 3
```

This reconstructs the state at wave 3, allowing you to:
- Modify the state programmatically
- Re-execute from that point
- Explore alternative execution paths

**Note**: State modifications must be done programmatically using the `CheckpointDebugger` API.

### Export to Archive

Export a run to a portable archive for sharing:

```bash
python -m pydantic_flow debug export <run_id> checkpoint.tar.gz
```

Output shows:
- Archive path
- Number of snapshots
- Archive size

Archives include:
- All state snapshots (binary msgpack)
- Execution traces (JSON)
- Run metadata (JSON)

### Import from Archive

Import a run from a shared archive:

```bash
python -m pydantic_flow debug import checkpoint.tar.gz

# Import to specific database
python -m pydantic_flow debug import checkpoint.tar.gz --db other.db
```

After import, you can use all debugging commands on the imported run.

## Common Workflows

### Debug a Failed Run

1. List runs to find the failed one:
   ```bash
   python -m pydantic_flow debug list-runs
   ```

2. View the timeline to see where it failed:
   ```bash
   python -m pydantic_flow debug timeline <run_id>
   ```

3. Replay to see detailed events:
   ```bash
   python -m pydantic_flow debug replay <run_id> --wave 5
   ```

4. Rewind to the wave before failure:
   ```bash
   python -m pydantic_flow debug rewind <run_id> --to-wave 4
   ```

### Compare Execution Paths

1. Export the first run:
   ```bash
   python -m pydantic_flow debug export <run_id_1> run1.tar.gz
   ```

2. Export the second run:
   ```bash
   python -m pydantic_flow debug export <run_id_2> run2.tar.gz
   ```

3. Import both to a comparison database:
   ```bash
   python -m pydantic_flow debug import run1.tar.gz --db compare.db
   python -m pydantic_flow debug import run2.tar.gz --db compare.db
   ```

4. Compare timelines:
   ```bash
   python -m pydantic_flow debug timeline <run_id_1> --db compare.db
   python -m pydantic_flow debug timeline <run_id_2> --db compare.db
   ```

### Share Debugging Session

1. Export the interesting run:
   ```bash
   python -m pydantic_flow debug export <run_id> debug_session.tar.gz
   ```

2. Share the archive file with teammates

3. Teammate imports and debugs:
   ```bash
   python -m pydantic_flow debug import debug_session.tar.gz
   python -m pydantic_flow debug details <run_id>
   ```

## Advanced: Programmatic Access

For complex debugging scenarios, use the Python API:

```python
from pathlib import Path
from pydantic_flow.checkpoints import CheckpointDebugger, SQLiteCheckpointBackend, SQLiteCheckpointConfig

# Create debugger
config = SQLiteCheckpointConfig(db_path=Path("checkpoints.db"))
backend = SQLiteCheckpointBackend(config=config)
debugger = CheckpointDebugger(backend=backend)

await backend.initialize()
try:
    # Fork with state modifications
    forked_state = await debugger.fork_from_wave(
        run_id="UYe_4hlpdJKt...",
        source_wave=3,
        state_modifications={
            "node_1": modified_state_model
        }
    )
    
    # Use forked_state to initialize new flow execution
    result = await my_flow.invoke(initial_state=forked_state, ...)
finally:
    await backend.close()
```

See `checkpoint_v2_basic_usage.py` for more examples.
