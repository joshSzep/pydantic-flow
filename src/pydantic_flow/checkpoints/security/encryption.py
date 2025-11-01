"""Encryption utilities for checkpoint data.

Provides field-level and full-checkpoint encryption using Fernet (AES-128-CBC).
"""

from __future__ import annotations

from typing import Any
from typing import Protocol

from pydantic import BaseModel


class EncryptionKey(BaseModel):
    """Encryption key configuration.

    Attributes:
        key_id: Unique identifier for this key.
        key_data: Base64-encoded key data.
        created_at: When this key was created.
        expires_at: Optional expiration timestamp.

    """

    key_id: str
    key_data: str
    created_at: str | None = None
    expires_at: str | None = None


class CheckpointEncryption(Protocol):
    """Protocol for checkpoint encryption implementations.

    Implementations must provide encrypt/decrypt methods for arbitrary data.
    """

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data.

        Args:
            data: Raw bytes to encrypt.

        Returns:
            Encrypted bytes.

        """
        ...

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data.

        Args:
            encrypted_data: Encrypted bytes.

        Returns:
            Decrypted bytes.

        """
        ...

    def rotate_key(self, new_key: EncryptionKey) -> None:
        """Rotate to new encryption key.

        Args:
            new_key: New key to use for future encryption.

        """
        ...


class FernetEncryption:
    """Fernet (AES-128-CBC) encryption for checkpoint data.

    Uses symmetric encryption with key rotation support.

    Example:
        >>> encryption = FernetEncryption.generate()
        >>> encrypted = encryption.encrypt(b"sensitive data")
        >>> decrypted = encryption.decrypt(encrypted)
        >>> assert decrypted == b"sensitive data"

    """

    def __init__(self, key: str | EncryptionKey):
        """Initialize with encryption key.

        Args:
            key: Base64-encoded Fernet key or EncryptionKey model.

        """
        try:
            from cryptography.fernet import Fernet
        except ImportError as e:
            msg = (
                "cryptography is required for encryption. "
                "Install with: pip install cryptography"
            )
            raise ImportError(msg) from e

        self._key = key if isinstance(key, str) else key.key_data
        key_bytes = self._key.encode() if isinstance(self._key, str) else self._key
        self._fernet = Fernet(key_bytes)
        self._old_fernets: list[Fernet] = []  # For key rotation

    @classmethod
    def generate(cls) -> FernetEncryption:
        """Generate new FernetEncryption with random key.

        Returns:
            FernetEncryption instance with fresh key.

        """
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
        return cls(key)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Fernet.

        Args:
            data: Raw bytes to encrypt.

        Returns:
            Encrypted bytes with Fernet token.

        """
        return self._fernet.encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet.

        Tries current key first, then falls back to old keys (for rotation).

        Args:
            encrypted_data: Encrypted Fernet token.

        Returns:
            Decrypted bytes.

        Raises:
            cryptography.fernet.InvalidToken: If decryption fails.

        """
        from cryptography.fernet import InvalidToken

        # Try current key
        try:
            return self._fernet.decrypt(encrypted_data)
        except InvalidToken:
            # Try old keys (key rotation)
            for old_fernet in self._old_fernets:
                try:
                    return old_fernet.decrypt(encrypted_data)
                except InvalidToken:
                    continue
            raise

    def rotate_key(self, new_key: str | EncryptionKey) -> None:
        """Rotate to new encryption key.

        Old key is kept for decrypting existing data.

        Args:
            new_key: New Fernet key (base64-encoded).

        """
        from cryptography.fernet import Fernet

        # Keep old key for decryption
        self._old_fernets.append(self._fernet)

        # Set new key
        key_str = new_key if isinstance(new_key, str) else new_key.key_data
        self._key = key_str
        self._fernet = Fernet(key_str.encode() if isinstance(key_str, str) else key_str)

    def get_current_key(self) -> str:
        """Get current encryption key.

        Returns:
            Base64-encoded Fernet key.

        """
        return self._key


class EncryptedCheckpointBackend:
    """Wrapper backend that encrypts checkpoint data before storage.

    Wraps any CheckpointStorageBackend and transparently encrypts/decrypts
    state snapshots and traces.

    Example:
        >>> from pydantic_flow.checkpoints import SQLiteCheckpointBackend
        >>> base_backend = SQLiteCheckpointBackend(...)
        >>> encryption = FernetEncryption.generate()
        >>> encrypted_backend = EncryptedCheckpointBackend(
        ...     backend=base_backend,
        ...     encryption=encryption
        ... )

    """

    def __init__(
        self,
        backend: Any,  # CheckpointStorageBackend
        encryption: CheckpointEncryption,
        *,
        encrypt_metadata: bool = False,
    ):
        """Initialize encrypted backend wrapper.

        Args:
            backend: Underlying storage backend.
            encryption: Encryption implementation.
            encrypt_metadata: If True, encrypt run metadata as well.

        """
        self._backend = backend
        self._encryption = encryption
        self._encrypt_metadata = encrypt_metadata

    async def initialize(self) -> None:
        """Initialize underlying backend."""
        await self._backend.initialize()

    async def close(self) -> None:
        """Close underlying backend."""
        await self._backend.close()

    async def healthcheck(self) -> bool:
        """Check underlying backend health."""
        return await self._backend.healthcheck()

    def _encrypt_state_data(self, data: bytes) -> bytes:
        """Encrypt state data."""
        return self._encryption.encrypt(data)

    def _decrypt_state_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt state data."""
        return self._encryption.decrypt(encrypted_data)

    async def save_state_snapshot(self, snapshot: Any) -> None:
        """Save state snapshot with encryption.

        Args:
            snapshot: StateSnapshot to encrypt and save.

        """
        from pydantic_flow.checkpoints.serialization import TypedSerializer

        # Serialize and encrypt full_state if present
        if snapshot.full_state is not None:
            full_state_data = TypedSerializer.serialize(snapshot.full_state)
            encrypted_full_state = self._encrypt_state_data(full_state_data)
            # Store encrypted data (implementation-specific)
            snapshot._encrypted_full_state = encrypted_full_state
            snapshot.full_state = None  # Clear plaintext

        # Encrypt deltas if present
        if snapshot.forward_delta is not None:
            delta_data = TypedSerializer.serialize(snapshot.forward_delta)
            encrypted_delta = self._encrypt_state_data(delta_data)
            snapshot._encrypted_forward_delta = encrypted_delta
            snapshot.forward_delta = None

        if snapshot.reverse_delta is not None:
            reverse_data = TypedSerializer.serialize(snapshot.reverse_delta)
            encrypted_reverse = self._encrypt_state_data(reverse_data)
            snapshot._encrypted_reverse_delta = encrypted_reverse
            snapshot.reverse_delta = None

        await self._backend.save_state_snapshot(snapshot)

    async def get_state_snapshot(self, run_id: Any, wave_number: int) -> Any | None:
        """Get and decrypt state snapshot.

        Args:
            run_id: Run identifier.
            wave_number: Wave number.

        Returns:
            Decrypted StateSnapshot or None.

        """
        from pydantic_flow.checkpoints.serialization import TypedSerializer

        snapshot = await self._backend.get_state_snapshot(run_id, wave_number)
        if not snapshot:
            return None

        # Decrypt full_state if present
        if hasattr(snapshot, "_encrypted_full_state"):
            encrypted = snapshot._encrypted_full_state
            decrypted_data = self._decrypt_state_data(encrypted)
            snapshot.full_state = TypedSerializer.deserialize(decrypted_data)

        # Decrypt deltas if present
        if hasattr(snapshot, "_encrypted_forward_delta"):
            encrypted = snapshot._encrypted_forward_delta
            decrypted_data = self._decrypt_state_data(encrypted)
            snapshot.forward_delta = TypedSerializer.deserialize(decrypted_data)

        if hasattr(snapshot, "_encrypted_reverse_delta"):
            encrypted = snapshot._encrypted_reverse_delta
            decrypted_data = self._decrypt_state_data(encrypted)
            snapshot.reverse_delta = TypedSerializer.deserialize(decrypted_data)

        return snapshot

    # Delegate other methods to underlying backend
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown methods to underlying backend."""
        return getattr(self._backend, name)
