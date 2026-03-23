"""
Test script for Telegram scheduler functionality
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from services.telegram_scheduler import telegram_scheduler, ScheduledTask, ScheduleType, OperationType


async def test_scheduler():
    """Test the scheduler with a sample NIFTY-50 price notification"""
    print("🚀 Testing Telegram Scheduler...")
    
    # Create a test task for NIFTY-50 price notification at 9:16 AM daily
    test_task = ScheduledTask(
        id="test_nifty_50",
        name="NIFTY-50 Morning Price Alert",
        description="Daily notification of NIFTY-50 current price at 9:16 AM",
        schedule_type=ScheduleType.DAILY,
        operation_type=OperationType.FETCH_PRICE,
        schedule_config={
            "time": "09:16"
        },
        operation_config={
            "symbol": "NIFTY-50"
        },
        enabled=True,
        timezone="Asia/Kolkata"
    )
    
    # Add the task
    try:
        success = await telegram_scheduler.add_task(test_task)
        
        if success:
            print("✅ Test task created successfully!")
            print(f"📅 Next run: {test_task.next_run}")
            
            # Start the scheduler
            await telegram_scheduler.start_scheduler()
            
            print("🔄 Scheduler started. Monitoring for next execution...")
            
            # Monitor for a few minutes
            for i in range(5):  # Monitor for 5 minutes
                await asyncio.sleep(60)  # Wait 1 minute
                
                # Check if task executed
                if test_task.last_run:
                    print(f"✅ Task executed at: {test_task.last_run}")
                    print(f"📅 Next run scheduled for: {test_task.next_run}")
                    break
            
            # Stop the scheduler
            await telegram_scheduler.stop_scheduler()
            print("⏹️ Scheduler stopped")
            
        else:
            print("❌ Failed to create test task")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")


if __name__ == "__main__":
    asyncio.run(test_scheduler())
