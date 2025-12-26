"""
Database repositories for data access operations
"""
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from database.connection import get_database
from database.models import AgentApproval, AgentLog, AgentConfig, SimulationResult, ToolExecution, ChatMessage


class AgentApprovalRepository:
    """Repository for agent approvals"""

    def save(self, approval: AgentApproval) -> bool:
        """Save an approval to database"""
        try:
            db = get_database()
            query = '''
                INSERT OR REPLACE INTO agent_approvals
                (approval_id, action, details, trade_value, risk_amount, reward_amount,
                 risk_percentage, rr_ratio, reasoning, status, created_at, approved_at,
                 rejected_at, approved_by, rejected_by, rejection_reason, symbol,
                 entry_price, quantity, stop_loss, target_price, entry_order_id, sl_order_id, tp_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''

            # Extract symbol and trade details from approval
            symbol = approval.details.get('symbol') if approval.details else None
            entry_price = approval.details.get('price') if approval.details else None
            quantity = approval.details.get('qty') if approval.details else None
            stop_loss = approval.details.get('sl') if approval.details else None
            target_price = approval.details.get('tp') if approval.details else None

            params = (
                approval.approval_id,
                approval.action,
                json.dumps(approval.details),
                approval.trade_value,
                approval.risk_amount,
                approval.reward_amount,
                approval.risk_percentage,
                approval.rr_ratio,
                approval.reasoning,
                approval.status,
                approval.created_at.isoformat(),
                approval.approved_at.isoformat() if approval.approved_at else None,
                approval.rejected_at.isoformat() if approval.rejected_at else None,
                approval.approved_by,
                approval.rejected_by,
                approval.rejection_reason,
                symbol,
                entry_price,
                quantity,
                stop_loss,
                target_price,
                approval.entry_order_id,
                approval.sl_order_id,
                approval.tp_order_id
            )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error saving approval: {e}")
            return False

    def get_by_id(self, approval_id: str) -> Optional[AgentApproval]:
        """Get approval by ID"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM agent_approvals WHERE approval_id = ?",
                (approval_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_approval(row)
        except Exception as e:
            print(f"Error getting approval: {e}")
        return None

    def get_pending(self) -> List[AgentApproval]:
        """Get all pending approvals"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM agent_approvals WHERE status = 'PENDING' ORDER BY created_at DESC"
            )
            return [self._row_to_approval(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting pending approvals: {e}")
            return []

    def get_approved(self) -> List[AgentApproval]:
        """Get all approved trades"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM agent_approvals WHERE status = 'APPROVED' ORDER BY approved_at DESC"
            )
            return [self._row_to_approval(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting approved trades: {e}")
            return []

    def get_all(self) -> List[AgentApproval]:
        """Get all approvals"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM agent_approvals ORDER BY created_at DESC"
            )
            return [self._row_to_approval(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting all approvals: {e}")
            return []

    def update_status(self, approval_id: str, status: str,
                     approved_by: Optional[str] = None,
                     rejected_by: Optional[str] = None,
                     rejection_reason: Optional[str] = None) -> bool:
        """Update approval status"""
        try:
            db = get_database()
            now = datetime.now().isoformat()

            if status == 'APPROVED':
                query = "UPDATE agent_approvals SET status = ?, approved_at = ?, approved_by = ? WHERE approval_id = ?"
                params = (status, now, approved_by, approval_id)
            elif status == 'REJECTED':
                query = "UPDATE agent_approvals SET status = ?, rejected_at = ?, rejected_by = ?, rejection_reason = ? WHERE approval_id = ?"
                params = (status, now, rejected_by, rejection_reason, approval_id)
            else:
                return False

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error updating approval status: {e}")
            return False

    def update_order_ids(self, approval_id: str, entry_order_id: Optional[str] = None,
                        sl_order_id: Optional[str] = None, tp_order_id: Optional[str] = None) -> bool:
        """Update order IDs for an approval"""
        try:
            db = get_database()
            updates = []
            params = []
            
            if entry_order_id is not None:
                updates.append("entry_order_id = ?")
                params.append(entry_order_id)
            if sl_order_id is not None:
                updates.append("sl_order_id = ?")
                params.append(sl_order_id)
            if tp_order_id is not None:
                updates.append("tp_order_id = ?")
                params.append(tp_order_id)
            
            if not updates:
                return False
            
            params.append(approval_id)
            query = f"UPDATE agent_approvals SET {', '.join(updates)} WHERE approval_id = ?"
            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error updating order IDs: {e}")
            return False

    def _row_to_approval(self, row) -> AgentApproval:
        """Convert database row to AgentApproval object"""
        return AgentApproval(
            approval_id=row['approval_id'],
            action=row['action'],
            details=json.loads(row['details']) if row['details'] else {},
            trade_value=row['trade_value'],
            risk_amount=row['risk_amount'],
            reward_amount=row['reward_amount'],
            risk_percentage=row['risk_percentage'],
            rr_ratio=row['rr_ratio'],
            reasoning=row['reasoning'] or '',
            status=row['status'],
            created_at=datetime.fromisoformat(row['created_at']),
            approved_at=datetime.fromisoformat(row['approved_at']) if row['approved_at'] else None,
            rejected_at=datetime.fromisoformat(row['rejected_at']) if row['rejected_at'] else None,
            approved_by=row['approved_by'],
            rejected_by=row['rejected_by'],
            rejection_reason=row['rejection_reason'],
            symbol=row['symbol'],
            entry_price=row['entry_price'],
            quantity=row['quantity'],
            stop_loss=row['stop_loss'],
            target_price=row['target_price'],
            entry_order_id=self._safe_get_column(row, 'entry_order_id'),
            sl_order_id=self._safe_get_column(row, 'sl_order_id'),
            tp_order_id=self._safe_get_column(row, 'tp_order_id')
        )
    
    def _safe_get_column(self, row, column_name):
        """Safely get a column value from a SQLite Row, returning None if column doesn't exist"""
        try:
            column_names = row.keys()
            if column_name in column_names:
                return row[column_name]
            return None
        except (KeyError, AttributeError):
            return None


class AgentLogRepository:
    """Repository for agent logs"""

    def save(self, log: AgentLog) -> bool:
        """Save a log entry"""
        try:
            db = get_database()
            query = '''
                INSERT INTO agent_logs (timestamp, level, message, component, metadata)
                VALUES (?, ?, ?, ?, ?)
            '''
            params = (
                log.timestamp.isoformat(),
                log.level,
                log.message,
                log.component,
                json.dumps(log.metadata) if log.metadata else None
            )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error saving log: {e}")
            return False

    def get_recent(self, limit: int = 100, component: Optional[str] = None) -> List[AgentLog]:
        """Get recent logs"""
        try:
            db = get_database()
            if component:
                cursor = db.execute_query(
                    "SELECT * FROM agent_logs WHERE component = ? ORDER BY timestamp DESC LIMIT ?",
                    (component, limit)
                )
            else:
                cursor = db.execute_query(
                    "SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )

            return [self._row_to_log(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting recent logs: {e}")
            return []

    def _row_to_log(self, row) -> AgentLog:
        """Convert database row to AgentLog object"""
        return AgentLog(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            level=row['level'],
            message=row['message'],
            component=row['component'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )


class AgentConfigRepository:
    """Repository for agent configuration"""

    def save(self, config: AgentConfig) -> bool:
        """Save configuration"""
        try:
            db = get_database()
            query = '''
                INSERT OR REPLACE INTO agent_config
                (key, value, value_type, category, description, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            params = (
                config.key,
                config.value,
                config.value_type,
                config.category,
                config.description,
                config.updated_at.isoformat()
            )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def get_by_key(self, key: str) -> Optional[AgentConfig]:
        """Get config by key"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM agent_config WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_config(row)
        except Exception as e:
            print(f"Error getting config: {e}")
        return None

    def get_by_category(self, category: str) -> List[AgentConfig]:
        """Get configs by category"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM agent_config WHERE category = ? ORDER BY key",
                (category,)
            )
            return [self._row_to_config(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting configs by category: {e}")
            return []

    def get_all(self) -> List[AgentConfig]:
        """Get all configurations"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM agent_config ORDER BY category, key"
            )
            return [self._row_to_config(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting all configs: {e}")
            return []

    def _row_to_config(self, row) -> AgentConfig:
        """Convert database row to AgentConfig object"""
        return AgentConfig(
            key=row['key'],
            value=row['value'],
            value_type=row['value_type'],
            category=row['category'],
            description=row['description'],
            updated_at=datetime.fromisoformat(row['updated_at'])
        )


class SimulationResultRepository:
    """Repository for simulation results"""

    def save(self, simulation: SimulationResult) -> bool:
        """Save simulation result"""
        try:
            db = get_database()
            query = '''
                INSERT OR REPLACE INTO simulation_results
                (simulation_id, instrument_name, date_range, strategy, trades, summary, created_at, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                simulation.simulation_id,
                simulation.instrument_name,
                simulation.date_range,
                simulation.strategy,
                json.dumps(simulation.trades),
                json.dumps(simulation.summary),
                simulation.created_at.isoformat(),
                simulation.file_path
            )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error saving simulation: {e}")
            return False

    def get_recent(self, limit: int = 10) -> List[SimulationResult]:
        """Get recent simulations"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM simulation_results ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_simulation(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting recent simulations: {e}")
            return []

    def _row_to_simulation(self, row) -> SimulationResult:
        """Convert database row to SimulationResult object"""
        return SimulationResult(
            simulation_id=row['simulation_id'],
            instrument_name=row['instrument_name'],
            date_range=row['date_range'],
            strategy=row['strategy'],
            trades=json.loads(row['trades']),
            summary=json.loads(row['summary']),
            created_at=datetime.fromisoformat(row['created_at']),
            file_path=row['file_path']
        )


class ToolExecutionRepository:
    """Repository for tool executions"""

    def save(self, execution: ToolExecution) -> bool:
        """Save tool execution"""
        try:
            db = get_database()
            query = '''
                INSERT INTO tool_executions
                (execution_id, tool_name, inputs, outputs, execution_time, success, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                execution.execution_id,
                execution.tool_name,
                json.dumps(execution.inputs),
                json.dumps(execution.outputs),
                execution.execution_time,
                execution.success,
                execution.error_message,
                execution.timestamp.isoformat()
            )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error saving tool execution: {e}")
            return False

    def get_recent(self, tool_name: Optional[str] = None, limit: int = 50) -> List[ToolExecution]:
        """Get recent tool executions"""
        try:
            db = get_database()
            if tool_name:
                cursor = db.execute_query(
                    "SELECT * FROM tool_executions WHERE tool_name = ? ORDER BY timestamp DESC LIMIT ?",
                    (tool_name, limit)
                )
            else:
                cursor = db.execute_query(
                    "SELECT * FROM tool_executions ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )

            return [self._row_to_execution(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting tool executions: {e}")
            return []

    def _row_to_execution(self, row) -> ToolExecution:
        """Convert database row to ToolExecution object"""
        return ToolExecution(
            execution_id=row['execution_id'],
            tool_name=row['tool_name'],
            inputs=json.loads(row['inputs']),
            outputs=json.loads(row['outputs']),
            execution_time=row['execution_time'],
            success=bool(row['success']),
            error_message=row['error_message'],
            timestamp=datetime.fromisoformat(row['timestamp'])
        )


class ChatMessageRepository:
    """Repository for chat messages"""

    def save(self, message: ChatMessage) -> bool:
        """Save chat message"""
        try:
            db = get_database()
            query = '''
                INSERT INTO chat_messages
                (message_id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            params = (
                message.message_id,
                message.session_id,
                message.role,
                message.content,
                message.timestamp.isoformat(),
                json.dumps(message.metadata) if message.metadata else None
            )

            db.execute_query(query, params)
            db.commit()
            return True
        except Exception as e:
            print(f"Error saving chat message: {e}")
            return False

    def get_session_messages(self, session_id: str, limit: int = 100) -> List[ChatMessage]:
        """Get messages for a session"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit)
            )
            return [self._row_to_message(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting session messages: {e}")
            return []

    def get_recent_messages(self, limit: int = 50) -> List[ChatMessage]:
        """Get recent messages across all sessions"""
        try:
            db = get_database()
            cursor = db.execute_query(
                "SELECT * FROM chat_messages ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_message(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting recent messages: {e}")
            return []

    def _row_to_message(self, row) -> ChatMessage:
        """Convert database row to ChatMessage object"""
        return ChatMessage(
            message_id=row['message_id'],
            session_id=row['session_id'],
            role=row['role'],
            content=row['content'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )


# Global repository instances
_approval_repo: Optional[AgentApprovalRepository] = None
_log_repo: Optional[AgentLogRepository] = None
_config_repo: Optional[AgentConfigRepository] = None
_simulation_repo: Optional[SimulationResultRepository] = None
_tool_repo: Optional[ToolExecutionRepository] = None
_chat_repo: Optional[ChatMessageRepository] = None


def get_approval_repository() -> AgentApprovalRepository:
    """Get approval repository instance"""
    global _approval_repo
    if _approval_repo is None:
        _approval_repo = AgentApprovalRepository()
    return _approval_repo


def get_log_repository() -> AgentLogRepository:
    """Get log repository instance"""
    global _log_repo
    if _log_repo is None:
        _log_repo = AgentLogRepository()
    return _log_repo


def get_config_repository() -> AgentConfigRepository:
    """Get config repository instance"""
    global _config_repo
    if _config_repo is None:
        _config_repo = AgentConfigRepository()
    return _config_repo


def get_simulation_repository() -> SimulationResultRepository:
    """Get simulation repository instance"""
    global _simulation_repo
    if _simulation_repo is None:
        _simulation_repo = SimulationResultRepository()
    return _simulation_repo


def get_tool_repository() -> ToolExecutionRepository:
    """Get tool execution repository instance"""
    global _tool_repo
    if _tool_repo is None:
        _tool_repo = ToolExecutionRepository()
    return _tool_repo


def get_chat_repository() -> ChatMessageRepository:
    """Get chat repository instance"""
    global _chat_repo
    if _chat_repo is None:
        _chat_repo = ChatMessageRepository()
    return _chat_repo
