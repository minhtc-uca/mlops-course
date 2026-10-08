def normalize_team_slug(team_name: str) -> str:
    if not team_name.strip():
        raise ValueError(
            "Team name must contain at least one non-whitespace character."
        )
    return team_name.strip().lower().replace(" ", "-")
