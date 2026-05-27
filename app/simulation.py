from __future__ import annotations

import asyncio
import random
import sys
import os

from app.agent.base import CognitiveAgent, Role
from app.environment.engine import Engine
from app.environment.session_manager import SessionManager


AGENT_IDS = ["Arjun", "Priya", "Rohan", "Meera"]


def assign_roles(agent_ids: list[str]) -> dict[str, Role]:
    roles = [Role.DETECTIVE, Role.ANALYST, Role.PSYCHOLOGIST, Role.SPY]
    random.shuffle(roles)
    return dict(zip(agent_ids, roles))


async def main():
    print("\nInitialising AgentRoom — Spy Among Agents\n")

    role_map = assign_roles(AGENT_IDS)
    spy_id   = next(aid for aid, role in role_map.items() if role == Role.SPY)

    agents = [
        CognitiveAgent(agent_id=aid, role=role_map[aid])
        for aid in AGENT_IDS
    ]

    session_manager = SessionManager()
    print("Registering agents:")
    for agent in agents:
        await session_manager.register_agent(agent)

    engine = Engine(
        agents=agents,
        thief_id=spy_id,
        session_manager=session_manager,
    )

    await engine.run()


if __name__ == "__main__":
    sys.stderr = open(os.devnull, 'w')
    asyncio.run(main())