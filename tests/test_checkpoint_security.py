"""Tests for checkpoint v2 security features.

Tests encryption, redaction, event filtering, GDPR compliance, and access control.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

from pydantic import BaseModel
import pytest

from pydantic_flow.checkpoints import SQLiteCheckpointBackend
from pydantic_flow.checkpoints import SQLiteCheckpointConfig
from pydantic_flow.checkpoints.security.access_control import AccessDeniedError
from pydantic_flow.checkpoints.security.access_control import AccessPolicy
from pydantic_flow.checkpoints.security.access_control import CheckpointAccessControl
from pydantic_flow.checkpoints.security.access_control import Permission
from pydantic_flow.checkpoints.security.access_control import create_readonly_backend
from pydantic_flow.checkpoints.security.encryption import FernetEncryption
from pydantic_flow.checkpoints.security.event_filter import EventFilter
from pydantic_flow.checkpoints.security.event_filter import EventFilterAction
from pydantic_flow.checkpoints.security.event_filter import EventFilterRule
from pydantic_flow.checkpoints.security.gdpr import GDPRErasureManager
from pydantic_flow.checkpoints.security.redaction import PIIPattern
from pydantic_flow.checkpoints.security.redaction import RedactionPolicy
from pydantic_flow.checkpoints.security.redaction import RedactionStrategy
from pydantic_flow.checkpoints.security.redaction import Redactor
from pydantic_flow.checkpoints.types import RunMetadata
from pydantic_flow.checkpoints.types import StateSnapshot
from pydantic_flow.checkpoints.types import generate_run_id
from pydantic_flow.checkpoints.types import generate_snapshot_id


class StoredEvent(BaseModel):
    """Test event model."""

    event_type: str
    timestamp: datetime
    data: dict[str, object]


class SimpleTestState(BaseModel):
    """Simple test state model."""

    data: str


class UserState(BaseModel):
    """Test state with user data."""

    user_email: str
    ssn: str
    phone: str


# =============================================================================
# Test 1: Encryption
# =============================================================================


@pytest.mark.asyncio
async def test_fernet_encryption_roundtrip():
    """Test Fernet encryption encrypt/decrypt roundtrip."""
    encryption = FernetEncryption.generate()

    original = b"sensitive data that needs protection"
    encrypted = encryption.encrypt(original)

    # Encrypted should be different
    assert encrypted != original
    assert len(encrypted) > len(original)

    # Decrypt should restore original
    decrypted = encryption.decrypt(encrypted)
    assert decrypted == original


@pytest.mark.asyncio
async def test_fernet_key_rotation():
    """Test key rotation with backward compatibility."""
    encryption = FernetEncryption.generate()

    # Encrypt with original key
    data = b"data encrypted with old key"
    encrypted_old = encryption.encrypt(data)

    # Rotate to new key
    new_key = FernetEncryption.generate().get_current_key()
    encryption.rotate_key(new_key)

    # Should still decrypt old data
    decrypted_old = encryption.decrypt(encrypted_old)
    assert decrypted_old == data

    # New encryptions use new key
    encrypted_new = encryption.encrypt(b"new data")
    decrypted_new = encryption.decrypt(encrypted_new)
    assert decrypted_new == b"new data"


# =============================================================================
# Test 2: Redaction
# =============================================================================


@pytest.mark.asyncio
async def test_redact_email_addresses():
    """Test redaction of email addresses from state."""
    policy = RedactionPolicy(
        enabled=True,
        patterns=[PIIPattern.EMAIL],
        strategy=RedactionStrategy.MASK,
    )
    redactor = Redactor(policy)

    state = {"user": {"email": "john@example.com", "name": "John"}}
    redacted = redactor.redact_state(state)

    # Email should be redacted
    assert "john@example.com" not in str(redacted)
    # Name should remain
    assert redacted["user"]["name"] == "John"


@pytest.mark.asyncio
async def test_redact_ssn_with_partial_strategy():
    """Test SSN redaction with partial visibility."""
    policy = RedactionPolicy(
        enabled=True,
        patterns=[PIIPattern.SSN],
        strategy=RedactionStrategy.PARTIAL,
    )
    redactor = Redactor(policy)

    text = "SSN: 123-45-6789"
    redacted = redactor.redact_string(text)

    # Should show partial (first 3 and last 4)
    assert "123" in redacted
    assert "6789" in redacted
    assert "45" not in redacted


@pytest.mark.asyncio
async def test_redact_field_names():
    """Test redaction by field name."""
    policy = RedactionPolicy(
        enabled=True,
        field_names=["password", "api_key"],
        strategy=RedactionStrategy.MASK,
    )
    redactor = Redactor(policy)

    state = {"user": {"password": "secret123", "api_key": "key_abc", "name": "Alice"}}
    redacted = redactor.redact_state(state)

    assert redacted["user"]["password"] == "***REDACTED***"
    assert redacted["user"]["api_key"] == "***REDACTED***"
    assert redacted["user"]["name"] == "Alice"


@pytest.mark.asyncio
async def test_redact_multiple_patterns():
    """Test redacting multiple PII patterns."""
    policy = RedactionPolicy(
        enabled=True,
        patterns=[PIIPattern.EMAIL, PIIPattern.PHONE],
        strategy=RedactionStrategy.MASK,
        preserve_length=True,
    )
    redactor = Redactor(policy)

    text = "Contact: john@example.com or call 555-123-4567"
    redacted = redactor.redact_string(text)

    # Both patterns should be redacted
    assert "john@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "Contact:" in redacted
    assert "or call" in redacted


# =============================================================================
# Test 3: Event Filtering
# =============================================================================


@pytest.mark.asyncio
async def test_event_filter_exclude_by_type():
    """Test filtering events by type."""
    rules = [
        EventFilterRule(
            name="exclude_tool_calls",
            event_type_pattern="tool_call",
            action=EventFilterAction.EXCLUDE,
        )
    ]
    event_filter = EventFilter(rules=rules)

    events = [
        StoredEvent(
            event_type="token",
            timestamp=datetime.now(UTC),
            data={"text": "hello"},
        ),
        StoredEvent(
            event_type="tool_call",
            timestamp=datetime.now(UTC),
            data={"tool_name": "search", "args": {}},
        ),
        StoredEvent(
            event_type="token",
            timestamp=datetime.now(UTC),
            data={"text": "world"},
        ),
    ]

    filtered = event_filter.filter_events(events)

    # Should only have 2 token events
    assert len(filtered) == 2
    assert all(e.event_type == "token" for e in filtered)


@pytest.mark.asyncio
async def test_event_filter_redact_fields():
    """Test redacting specific fields from events."""
    rules = [
        EventFilterRule(
            name="redact_api_keys",
            tool_name_pattern=".*api.*",
            action=EventFilterAction.REDACT,
            redact_fields=["api_key", "auth_token"],
        )
    ]
    event_filter = EventFilter(rules=rules)

    events = [
        StoredEvent(
            event_type="tool_call",
            timestamp=datetime.now(UTC),
            data={
                "tool_name": "call_api",
                "api_key": "secret123",
                "auth_token": "token456",
                "endpoint": "/users",
            },
        )
    ]

    filtered = event_filter.filter_events(events)

    assert len(filtered) == 1
    assert filtered[0].data["api_key"] == "***REDACTED***"
    assert filtered[0].data["auth_token"] == "***REDACTED***"
    assert filtered[0].data["endpoint"] == "/users"


@pytest.mark.asyncio
async def test_event_filter_priority():
    """Test rule priority ordering."""
    rules = [
        EventFilterRule(
            name="allow_all",
            event_type_pattern=".*",
            action=EventFilterAction.ALLOW,
            priority=0,
        ),
        EventFilterRule(
            name="exclude_sensitive",
            event_type_pattern="sensitive_.*",
            action=EventFilterAction.EXCLUDE,
            priority=10,  # Higher priority
        ),
    ]
    event_filter = EventFilter(rules=rules)

    events = [
        StoredEvent(
            event_type="normal_event",
            timestamp=datetime.now(UTC),
            data={},
        ),
        StoredEvent(
            event_type="sensitive_event",
            timestamp=datetime.now(UTC),
            data={},
        ),
    ]

    filtered = event_filter.filter_events(events)

    # Higher priority exclude rule should win
    assert len(filtered) == 1
    assert filtered[0].event_type == "normal_event"


# =============================================================================
# Test 4: GDPR Compliance
# =============================================================================


@pytest.mark.asyncio
async def test_gdpr_search_user_data(tmp_path):
    """Test searching for user data in checkpoints."""
    # Create backend with test data
    db_path = tmp_path / "gdpr_test.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        run_id = generate_run_id()

        # Save metadata
        metadata = RunMetadata(
            run_id=run_id,
            flow_id="test_flow",
            started_at=datetime.now(UTC),
            status=RunMetadata.Status.COMPLETED,
            total_waves=1,
        )
        await backend.save_run_metadata(metadata)

        # Save checkpoint with user data
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={
                "user": UserState(
                    user_email="john@example.com",
                    ssn="123-45-6789",
                    phone="555-1234",
                )
            },
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )
        await backend.save_state_snapshot(snapshot)

        # Search for user data
        gdpr = GDPRErasureManager(backend)
        results = await gdpr.search_user_data("john@example.com")

        assert len(results) > 0
        assert any("user_email" in r.field_paths[0] for r in results)

    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_gdpr_erase_user_data(tmp_path):
    """Test erasing user data (dry run)."""
    db_path = tmp_path / "gdpr_erase_test.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    backend = SQLiteCheckpointBackend(config)
    await backend.initialize()

    try:
        run_id = generate_run_id()

        # Save metadata
        metadata = RunMetadata(
            run_id=run_id,
            flow_id="test_flow",
            started_at=datetime.now(UTC),
            status=RunMetadata.Status.COMPLETED,
            total_waves=1,
        )
        await backend.save_run_metadata(metadata)

        # Save checkpoint with user data
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={
                "user": UserState(
                    user_email="alice@example.com",
                    ssn="987-65-4321",
                    phone="555-9999",
                )
            },
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )
        await backend.save_state_snapshot(snapshot)

        # Erase user data (dry run)
        gdpr = GDPRErasureManager(backend)
        log = await gdpr.erase_user_data(
            "alice@example.com",
            operator="admin@company.com",
            dry_run=True,
        )

        assert log.user_identifier == "alice@example.com"
        assert log.waves_affected > 0
        assert log.operator == "admin@company.com"

    finally:
        await backend.close()


# =============================================================================
# Test 5: Access Control
# =============================================================================


@pytest.mark.asyncio
async def test_access_control_read_permission(tmp_path):
    """Test read-only access control."""
    db_path = tmp_path / "access_test.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    base_backend = SQLiteCheckpointBackend(config)
    await base_backend.initialize()

    try:
        # Create readonly backend
        readonly_backend = create_readonly_backend(base_backend, "analyst")

        run_id = generate_run_id()

        # Admin saves data (using base backend)
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"state": SimpleTestState(data="test")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )
        await base_backend.save_state_snapshot(snapshot)

        # Readonly user can read
        retrieved = await readonly_backend.get_state_snapshot(run_id, 0)
        assert retrieved is not None

        # Readonly user cannot write
        new_snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=1,
            full_state={"state": SimpleTestState(data="new")},
            state_hash="hash2",
            next_frontier=[],
            routing_ended=False,
        )

        with pytest.raises(AccessDeniedError):
            await readonly_backend.save_state_snapshot(new_snapshot)

    finally:
        await base_backend.close()


@pytest.mark.asyncio
async def test_access_control_run_filtering(tmp_path):
    """Test filtering runs by access control."""
    db_path = tmp_path / "filter_test.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    base_backend = SQLiteCheckpointBackend(config)
    await base_backend.initialize()

    try:
        # Create two runs
        run_id_1 = generate_run_id()
        run_id_2 = generate_run_id()

        # Save metadata for both
        for run_id in [run_id_1, run_id_2]:
            metadata = RunMetadata(
                run_id=run_id,
                flow_id="test",
                started_at=datetime.now(UTC),
                status=RunMetadata.Status.COMPLETED,
                total_waves=0,
            )
            await base_backend.save_run_metadata(metadata)

        # Create policy that only allows access to run 1
        policy = AccessPolicy(
            user_id="user1",
            permissions=[Permission.READ_CHECKPOINT],
            allowed_run_ids=[str(run_id_1)],
        )

        protected_backend = CheckpointAccessControl(
            backend=base_backend,
            policies={"user1": policy},
            current_user="user1",
        )

        # List runs should only return run 1
        runs = await protected_backend.list_runs()
        assert len(runs) == 1
        assert runs[0].run_id == run_id_1

    finally:
        await base_backend.close()


@pytest.mark.asyncio
async def test_access_control_admin_permission(tmp_path):
    """Test admin permission grants all access."""
    db_path = tmp_path / "admin_test.db"
    config = SQLiteCheckpointConfig(db_path=db_path)
    base_backend = SQLiteCheckpointBackend(config)
    await base_backend.initialize()

    try:
        run_id = generate_run_id()

        # Create admin policy
        policy = AccessPolicy(
            user_id="admin",
            permissions=[Permission.ADMIN],
        )

        admin_backend = CheckpointAccessControl(
            backend=base_backend,
            policies={"admin": policy},
            current_user="admin",
        )

        # Admin can do everything
        snapshot = StateSnapshot(
            snapshot_id=generate_snapshot_id(),
            run_id=run_id,
            wave_number=0,
            full_state={"state": SimpleTestState(data="test")},
            state_hash="hash",
            next_frontier=[],
            routing_ended=False,
        )

        # Should not raise
        await admin_backend.save_state_snapshot(snapshot)
        retrieved = await admin_backend.get_state_snapshot(run_id, 0)
        assert retrieved is not None

    finally:
        await base_backend.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
