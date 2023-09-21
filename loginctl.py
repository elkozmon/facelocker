import subprocess
import sys

class Session:
    def __init__(self, id: str, user: str, locked: bool):
        self._id = id
        self._user = user
        self._locked = locked

    def __str__(self):
        return f"Session(id={self._id}, user={self._user}, locked={self._locked})"

    def __repr__(self):
        return str(self)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def user(self) -> str:
        return self._user
    
    @property
    def locked(self) -> bool:
        return self._locked

class LoginCtl:
    def __init__(self):
        pass

    def list_user_sessions(self, user: str, unlocked_only: bool = False) -> list[Session]:
        sessions = self._list_sessions()
        user_sessions = []

        for session in sessions:
            if session.user != user:
                continue

            if unlocked_only and session.locked:
                continue

            user_sessions.append(session)

        return user_sessions

    def list_unlocked_sessions(self) -> list[Session]:
        sessions = self._list_sessions()
        unlocked_sessions = []

        for session in sessions:
            if session.locked:
                continue

            unlocked_sessions.append(session)

        return unlocked_sessions

    def lock_sessions(self, sessions: list[Session]):
        subprocess.check_call(["loginctl", "lock-session"] + [session.id for session in sessions])

    def lock_session(self, session: Session):
        subprocess.check_call(["loginctl", "lock-session", session.id])

    def _list_sessions(self) -> list[Session]:
        output = subprocess.check_output(["loginctl", "list-sessions", "--no-legend"]).decode("utf-8")
        lines = output.strip().split("\n")
        sessions = []

        for line in lines:
            fields = line.strip().split()
            session_id = fields[0]
            session_user = fields[2]
            session_locked = self._is_session_locked(session_id)

            sessions.append(Session(session_id, session_user, session_locked))

        return sessions

    def _is_session_locked(self, session_id: str) -> bool:
        output = subprocess.check_output(["loginctl", "show-session", session_id]).decode("utf-8")
        lines = output.strip().split("\n")

        for line in lines:
            if line.startswith("LockedHint="):
                locked_hint = line.split("=")[1]

                return locked_hint == "yes"

        sys.exit("unable to get session lock status")
