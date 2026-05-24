"""
Parse Kite Connect user/margins equity segment into available, utilised, and net values.
"""


def parse_equity_margins(equity_data: dict) -> tuple[float, float, float]:
    """
    Extract available, utilised, and net margin from Kite equity segment.

    Kite stores tradable funds in available.live_balance (not always in available.cash).
    See https://kite.trade/docs/connect/v3/user/#margins
    """
    available_value = equity_data.get("available", 0)
    utilised_value = equity_data.get("utilised", 0)

    if isinstance(available_value, dict):
        available_margin = (
            available_value.get("live_balance")
            or available_value.get("opening_balance")
            or available_value.get("cash")
            or 0
        )
    else:
        available_margin = available_value or 0

    if isinstance(utilised_value, dict):
        utilised_margin = utilised_value.get("debits", 0)
    else:
        utilised_margin = utilised_value or 0

    total_margin = equity_data.get("net", 0) or available_margin

    return (
        float(available_margin or 0),
        float(utilised_margin or 0),
        float(total_margin or 0),
    )
