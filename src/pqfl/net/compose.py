"""Generate the campaign compose file: one container per site plus the reducer."""
from pqfl.runner import site_ids


def service_name(actor: str) -> str:
    return actor.lower()


def campaign_compose(max_sites: int, *, image: str = "pqfl-crypto") -> dict:
    """Containers are not CPU-pinned: the replay is I/O bound and compute is measured
    separately in the simulator. The volume is the parent of the compose directory
    (the repository's `runs/`), mounted at /ledgers for the controller."""
    services = {}
    for actor in [*site_ids(max_sites), "reducer"]:
        services[service_name(actor)] = {
            "image": image, "cap_add": ["NET_ADMIN"],
            "environment": {"ACTOR": actor, "NETEM_PROFILE": "${NETEM_PROFILE:-lan}"}}
    services["controller"] = {"image": image, "profiles": ["tools"], "volumes": ["../:/ledgers"]}
    return {"services": services}


def agent_arguments(actors: list[str], port: int = 9000) -> list[str]:
    return [f"{actor}={service_name(actor)}:{port}" for actor in actors]
