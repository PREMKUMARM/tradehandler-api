"""
Approval queue management for hybrid trading mode
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from agent.config import get_agent_config
from agent.safety import get_safety_manager
from database.repositories import get_approval_repository, AgentApproval
from database.models import AgentApproval as ApprovalModel


class ApprovalQueue:
    """Manages approval queue for trades requiring human approval"""

    def __init__(self):
        self.config = get_agent_config()
        self.safety = get_safety_manager()
        self.on_create_callback = None
        self.repo = get_approval_repository()
    
    def needs_approval(self, trade_value: float, risk_amount: float, trade_type: str = "ORDER") -> bool:
        """Check if a trade needs approval"""
        # If auto-trade is globally enabled from config, nothing needs approval
        if self.config.is_auto_trade_enabled:
            return False

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
        reward_amount: float = 0.0,
        reasoning: str = ""
    ) -> str:
        """Create an approval request"""
        approval_id = str(uuid.uuid4())
        now = datetime.now()

        # Calculate risk metrics
        risk_percentage = (risk_amount / trade_value) * 100 if trade_value > 0 else 0
        rr_ratio = (reward_amount / risk_amount) if risk_amount > 0 else 0

        # Create approval model
        approval_model = ApprovalModel(
            approval_id=approval_id,
            action=action,
            details=details,
            trade_value=trade_value,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_percentage=risk_percentage,
            rr_ratio=rr_ratio,
            reasoning=reasoning,
            status="PENDING",
            created_at=now
        )

        # Save to database
        success = self.repo.save(approval_model)
        if not success:
            raise Exception("Failed to save approval to database")

        # Convert to dict for callback compatibility
        approval_dict = approval_model.model_dump()

        # Trigger callback if set
        if self.on_create_callback:
            try:
                self.on_create_callback(approval_dict)
            except Exception as e:
                print(f"Error in approval callback: {e}")

        return approval_id
    
    def get_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """Get approval by ID"""
        approval = self.repo.get_by_id(approval_id)
        return approval.model_dump() if approval else None

    def list_pending(self) -> List[Dict[str, Any]]:
        """List all pending approvals"""
        approvals = self.repo.get_pending()
        return [approval.model_dump() for approval in approvals]

    def approve(self, approval_id: str, approved_by: str = "user") -> bool:
        """Approve a pending action"""
        return self.repo.update_status(approval_id, "APPROVED", approved_by=approved_by)

    def reject(self, approval_id: str, reason: str = "", rejected_by: str = "user") -> bool:
        """Reject a pending action"""
        return self.repo.update_status(approval_id, "REJECTED", rejected_by=rejected_by, rejection_reason=reason)
    
    def list_approved(self) -> List[Dict[str, Any]]:
        """List all approved trades"""
        approvals = self.repo.get_approved()
        return [approval.model_dump() for approval in approvals]

    def get_stats(self) -> Dict[str, Any]:
        """Get approval queue statistics"""
        all_approvals = self.repo.get_all()
        pending = len([a for a in all_approvals if a.status == "PENDING"])
        approved = len([a for a in all_approvals if a.status == "APPROVED"])
        rejected = len([a for a in all_approvals if a.status == "REJECTED"])

        return {
            "total": len(all_approvals),
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

