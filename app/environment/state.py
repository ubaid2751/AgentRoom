from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    agent_id: str
    content:  str
    round:    int

    def to_log_line(self) -> str:
        return f"[Round {self.round}] {self.agent_id}: {self.content}"


@dataclass
class EnvironmentSnapshot:
    """
    Full world view passed to every agent before they think.
    No phase gating — agents see everything and decide freely.
    Created once per round, never mutated after creation.
    """
    round:             int
    active_agents:     list[str]
    eliminated_agents: list[str]
    global_log:        str
    vote_tally:        dict[str, int]
    accusation_log:    list[dict]


@dataclass
class WorldState:
    current_round:     int               = 0
    active_agents:     list[str]         = field(default_factory=list)
    eliminated_agents: list[str]         = field(default_factory=list)
    conversation_log:  list[Message]     = field(default_factory=list)
    accusation_log:    list[dict]        = field(default_factory=list)
    vote_tally:        dict[str, int]    = field(default_factory=dict)
    winner:            Optional[str]     = None

    def build_global_log(self) -> str:
        if not self.conversation_log:
            return "No conversation yet. This is the start of the game."
        return "\n".join(m.to_log_line() for m in self.conversation_log)

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            round=self.current_round,
            active_agents=list(self.active_agents),
            eliminated_agents=list(self.eliminated_agents),
            global_log=self.build_global_log(),
            vote_tally=dict(self.vote_tally),
            accusation_log=list(self.accusation_log),
        )

    def add_message(self, agent_id: str, speech: str):
        self.conversation_log.append(
            Message(agent_id=agent_id, content=speech, round=self.current_round)
        )

    def record_accusation(self, accuser: str, target: str):
        self.accusation_log.append({
            "accuser": accuser,
            "target":  target,
            "round":   self.current_round,
        })

    def cast_vote(self, target: str):
        self.vote_tally[target] = self.vote_tally.get(target, 0) + 1

    def resolve_votes(self) -> Optional[str]:
        """Return agent_id with most votes, or None if tied."""
        if not self.vote_tally:
            return None
        max_votes = max(self.vote_tally.values())
        candidates = [a for a, v in self.vote_tally.items() if v == max_votes]
        return candidates[0] if len(candidates) == 1 else None

    def eliminate(self, agent_id: str):
        if agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
            self.eliminated_agents.append(agent_id)

    def reset_votes(self):
        self.vote_tally.clear()
