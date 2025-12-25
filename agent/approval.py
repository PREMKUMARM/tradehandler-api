"""
Approval queue management for hybrid trading mode
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from agent.config import get_agent_config
from agent.safety import get_safety_manager


class ApprovalQueue:
    """Manages approval queue for trades requiring human approval"""
    
    def __init__(self):
        self.queue: Dict[str, Dict[str, Any]] = {}
        self.config = get_agent_config()
        self.safety = get_safety_manager()
    
    def needs_approval(self, trade_value: float, risk_amount: float, trade_type: str = "ORDER") -> bool:
        """Check if a trade needs approval"""
        # Auto-approve if below threshold
        if trade_value <= self.config.auto_trade_threshold:
            return False
        
        # Check risk percentage
        risk_pct = (risk_amount / trade_value) * 100 if trade_value > 0 else 0
        if risk_pct > self.config.risk_per_trade_pct * 2:  # Double the normal risk
            return True
        
        # Check if exceeds position size
        if trade_value > self.config.max_position_size * 0.8:  # 80% of max
            return True
        
        return False
    
    def create_approval(
        self,
        action: str,
        details: Dict[str, Any],
        trade_value: float,
        risk_amount: float,
        reasoning: str = ""
    ) -> str:
        """Create an approval request"""
        approval_id = str(uuid.uuid4())
        
        approval = {
            "approval_id": approval_id,
            "action": action,
            "details": details,
            "trade_value": trade_value,
            "risk_amount": risk_amount,
            "risk_percentage": (risk_amount / trade_value) * 100 if trade_value > 0 else 0,
            "reasoning": reasoning,
            "status": "PENDING",
            "created_at": datetime.now().isoformat(),
            "approved_at": None,
            "rejected_at": None,
            "approved_by": None
        }
        
        self.queue[approval_id] = approval
        return approval_id
    
    def get_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Get approval by ID"""
        return self.queue.get(approval_id)
    
    def list_pending(self) -> List[Dict[str, Any]]:
        """List all pending approvals"""
        return [
            approval for approval in self.queue.values()
            if approval["status"] == "PENDING"
        ]
    
    def approve(self, approval_id: str, approved_by: str = "user") -> bool:
        """Approve a pending action"""
        approval = self.queue.get(approval_id)
        if not approval:
            return False
        
        if approval["status"] != "PENDING":
            return False
        
        approval["status"] = "APPROVED"
        approval["approved_at"] = datetime.now().isoformat()
        approval["approved_by"] = approved_by
        
        return True
    
    def reject(self, approval_id: str, reason: str = "", rejected_by: str = "user") -> bool:
        """Reject a pending action"""
        approval = self.queue.get(approval_id)
        if not approval:
            return False
        
        if approval["status"] != "PENDING":
            return False
        
        approval["status"] = "REJECTED"
        approval["rejected_at"] = datetime.now().isoformat()
        approval["rejected_by"] = rejected_by
        approval["rejection_reason"] = reason
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get approval queue statistics"""
        pending = len([a for a in self.queue.values() if a["status"] == "PENDING"])
        approved = len([a for a in self.queue.values() if a["status"] == "APPROVED"])
        rejected = len([a for a in self.queue.values() if a["status"] == "REJECTED"])
        
        return {
            "total": len(self.queue),
            "pending": pending,
            "approved": approved,
            "rejected": rejected
        }


# Global approval queue instance
_approval_queue: Optional[ApprovalQueue] = None


def get_approval_queue() -> ApprovalQueue:
    """Get or create approval queue instance"""
    global _approval_queue
    if _approval_queue is None:
        _approval_queue = ApprovalQueue()
    return _approval_queue

