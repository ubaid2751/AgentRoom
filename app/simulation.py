from __future__ import annotations

import asyncio
import logging
import random

from app.agent.base import CognitiveAgent, Role
from app.environment.engine import Engine
from app.environment.session_manager import SessionManager

import logging

logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("google.adk").setLevel(logging.CRITICAL)
logging.getLogger("google.adk.runners").setLevel(logging.CRITICAL)


AGENT_IDS = ["Arjun", "Priya", "Rohan", "Meera"]


def assign_roles(agent_ids: list[str]) -> dict[str, Role]:
    roles = [Role.KING, Role.MINISTER, Role.POLICE, Role.THIEF]
    random.shuffle(roles)
    return dict(zip(agent_ids, roles))


async def main():
    print("\nInitialising simulation...")

    role_map = assign_roles(AGENT_IDS)
    thief_id = next(aid for aid, role in role_map.items() if role == Role.THIEF)

    agents = [
        CognitiveAgent(agent_id=aid, role=role_map[aid])
        for aid in AGENT_IDS
    ]

    session_manager = SessionManager()
    print("\nRegistering agents:")
    for agent in agents:
        await session_manager.register_agent(agent)

    engine = Engine(
        agents=agents,
        thief_id=thief_id,
        session_manager=session_manager,
    )

    await engine.run()


if __name__ == "__main__":
    import sys, os
    sys.stderr = open(os.devnull, 'w')
    asyncio.run(main())