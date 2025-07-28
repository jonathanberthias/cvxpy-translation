import pytest
from pytest_insta import SnapshotSession


class NoDeletionSnapshotSession(SnapshotSession):
    """A snapshot session that does not delete snapshots."""

    @property
    def should_delete(self) -> bool:
        return False


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config._no_deletion_snapshot_session = NoDeletionSnapshotSession(session)  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
