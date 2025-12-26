"""
Autonomous background scanner for the AI Agent.
Runs periodically in the background to identify signals and create approvals.
"""
import asyncio
from datetime import datetime, time
import traceback
from agent.config import get_agent_config
from agent.tools.trading_opportunities_tool import find_indicator_based_trading_opportunities
from agent.approval import get_approval_queue
from agent.tools.instrument_resolver import INSTRUMENT_GROUPS
from agent.ws_manager import broadcast_agent_update, add_agent_log

# Redundant add_autonomous_log removed - using agent.ws_manager.add_agent_log

async def run_autonomous_scan():
    """Periodically scan the market for opportunities"""
    config = get_agent_config()
    print(f"[AUTONOMOUS] Background scanner initializing...")
    
    # Track last scan time to respect interval
    last_scan_time = None
    
    while True:
        try:
            # Check if autonomous mode is enabled
            if not config.autonomous_mode:
                await asyncio.sleep(60)
                continue

            now = datetime.now()
            curr_time = now.time()
            
            # Respect market hours (09:15 - 15:30 IST)
            # However, for signals we use our Prime Session filter (10:15 - 14:45)
            # which is internally handled by the tool.
            is_market_open = time(9, 15) <= curr_time <= time(15, 30)
            
            if not is_market_open:
                # Outside market hours, sleep longer
                await asyncio.sleep(300)
                continue

            # Respect scan interval
            if last_scan_time:
                elapsed = (now - last_scan_time).total_seconds() / 60
                if elapsed < config.autonomous_scan_interval_mins:
                    await asyncio.sleep(30)
                    continue

            add_agent_log(f"Starting autonomous scan for {config.autonomous_target_group}")
            
            # Resolve the group instruments
            instruments = INSTRUMENT_GROUPS.get(config.autonomous_target_group, [])
            if not instruments:
                add_agent_log(f"Target group '{config.autonomous_target_group}' not found.", "warning")
                await asyncio.sleep(60)
                continue

            # Run the strategy tool on live data
            result = find_indicator_based_trading_opportunities.invoke({
                "instrument_name": instruments,
                "interval": "5minute",
                "target_date": now.strftime("%Y-%m-%d"),
                "is_sequential": True
            })

            if result.get("status") == "success":
                opps = result.get("opportunities", [])
                add_agent_log(f"Scan complete. Found {len(opps)} total opportunities.")
            else:
                add_agent_log(f"Scan failed: {result.get('error', 'Unknown error')}", "error")

            last_scan_time = now
            
        except Exception as e:
            print(f"[AUTONOMOUS] Error in background scanner: {str(e)}")
            traceback.print_exc()
            await asyncio.sleep(60)
        
        # Poll every 30 seconds to check for config changes
        await asyncio.sleep(30)

def start_autonomous_agent():
    """Start the autonomous scanner in a background task"""
    asyncio.create_task(run_autonomous_scan())

