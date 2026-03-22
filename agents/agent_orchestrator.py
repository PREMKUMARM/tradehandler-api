"""
Agent Orchestrator - Manages and coordinates multiple agents
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import json

from .base_agent import BaseAgent, AgentTask, AgentStatus, AgentConfig
from .agent_types import AgentType, get_agent_config, get_agent_dependencies, AGENT_CONFIGURATIONS
from .communication import AgentCommunicationLayer, MessageType
from .premarket_agent import PreMarketAgent
from .market_agent import MarketAgent
from .portfolio_agent import PortfolioAgent
from .strategy_agent import StrategyAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent
from .monitoring_agent import MonitoringAgent
from services.telegram_service import telegram_service
from .premium_strategy_agent import PremiumStrategyAgent
from storage.agent_storage import AgentDataStore
from utils.logger import log_info, log_error, log_warning, log_debug

class AgentOrchestrator:
    """Orchestrates multiple agents in the system"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.communication_layer = AgentCommunicationLayer()
        self.data_store = AgentDataStore()
        self.running = False
        self.agent_classes = {
            AgentType.PREMARKET_ANALYZER: PreMarketAgent,
            AgentType.MARKET_ANALYZER: MarketAgent,
            AgentType.PORTFOLIO_MANAGER: PortfolioAgent,
            AgentType.STRATEGY_BUILDER: StrategyAgent,
            AgentType.STRATEGY_EXECUTOR: StrategyAgent,
            AgentType.RISK_MANAGER: RiskAgent,
            AgentType.ORDER_EXECUTOR: ExecutionAgent,
            AgentType.MONITORING_AGENT: MonitoringAgent,
            AgentType.PREMIUM_STRATEGY_AGENT: PremiumStrategyAgent
        }
        
        # Task queue for orchestrator
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, AgentTask] = {}
        
        # Performance metrics
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.start_time = None
    
    async def initialize(self):
        """Initialize the orchestrator and all agents"""
        
        log_info("Initializing Agent Orchestrator")
        
        # Initialize data store
        await self.data_store.initialize()
        
        # Initialize communication layer
        await self.communication_layer.initialize()
        
        # Create and initialize agents
        await self._create_agents()
        
        # Start orchestrator tasks
        self.running = True
        self.start_time = datetime.now()
        
        # Start background tasks
        asyncio.create_task(self._process_tasks())
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._coordinate_agent_communication())
        
        log_info("Agent Orchestrator initialized successfully")
    
    async def _create_agents(self):
        """Create and initialize all agents"""
        
        for agent_type, config in AGENT_CONFIGURATIONS.items():
            if not config.enabled:
                log_info(f"Agent {agent_type.value} is disabled, skipping")
                continue
            
            try:
                # Get agent class
                agent_class = self.agent_classes.get(agent_type)
                if not agent_class:
                    log_error(f"No agent class found for {agent_type.value}")
                    continue
                
                # Create agent instance
                agent = agent_class(
                    agent_id=config.agent_id,
                    name=config.name,
                    **config.__dict__
                )
                
                # Initialize agent
                await agent.initialize(self.communication_layer, self.data_store)
                
                # Store agent
                self.agents[config.agent_id] = agent
                
                log_info(f"Agent {config.name} created and initialized")
                
            except Exception as e:
                log_error(f"Failed to create agent {agent_type.value}: {e}")
    
    async def start_all_agents(self):
        """Start all agents"""
        
        log_info("Starting all agents")
        
        # Send Telegram notification
        await telegram_service.notify_system_event(
            event_type="Agent System Start",
            message="Multi-Agent Trading System is starting up...",
            priority="normal"
        )
        
        # Start agents in dependency order
        start_order = self._get_agent_start_order()
        
        for agent_id in start_order:
            if agent_id in self.agents:
                try:
                    await self.agents[agent_id].start()
                    log_info(f"Agent {agent_id} started")
                    
                    # Send individual agent start notification
                    agent_name = self.agents[agent_id].config.name
                    await telegram_service.notify_agent_status(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        status="started",
                        details="Agent successfully initialized and started"
                    )
                    
                except Exception as e:
                    log_error(f"Failed to start agent {agent_id}: {e}")
                    
                    # Send error notification
                    await telegram_service.notify_error(
                        error_type="Agent Start Failed",
                        error_message=str(e),
                        context=f"Agent ID: {agent_id}"
                    )
    
    async def stop_all_agents(self):
        """Stop all agents"""
        
        log_info("Stopping all agents")
        
        # Send Telegram notification
        await telegram_service.notify_system_event(
            event_type="Agent System Stop",
            message="Multi-Agent Trading System is shutting down...",
            priority="normal"
        )
        
        for agent_id, agent in self.agents.items():
            try:
                await agent.stop()
                log_info(f"Agent {agent_id} stopped")
                
                # Send individual agent stop notification
                agent_name = agent.config.name
                await telegram_service.notify_agent_status(
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status="stopped",
                    details="Agent successfully stopped"
                )
                
            except Exception as e:
                log_error(f"Failed to stop agent {agent_id}: {e}")
                
                # Send error notification
                await telegram_service.notify_error(
                    error_type="Agent Stop Failed",
                    error_message=str(e),
                    context=f"Agent ID: {agent_id}"
                )
            except Exception as e:
                log_error(f"Failed to stop agent {agent_id}: {e}")
        
        self.running = False
    
    def _get_agent_start_order(self) -> List[str]:
        """Get agent start order based on dependencies"""
        
        # Simple topological sort for dependencies
        start_order = []
        visited = set()
        
        def visit_agent(agent_id: str):
            if agent_id in visited:
                return
            
            visited.add(agent_id)
            
            # Visit dependencies first
            agent = self.agents.get(agent_id)
            if agent:
                for dep_agent_id in agent.config.dependencies:
                    if dep_agent_id in self.agents:
                        visit_agent(dep_agent_id)
            
            start_order.append(agent_id)
        
        # Visit all agents
        for agent_id in self.agents:
            visit_agent(agent_id)
        
        return start_order
    
    async def process_user_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process user request and route to appropriate agents"""
        
        request_type = request.get("request_type", "general")
        user_message = request.get("message", "")
        user_context = request.get("context", {})
        
        log_info(f"Processing user request: {request_type}")
        
        try:
            # Analyze request to determine which agents to use
            agent_selection = await self._analyze_request(request)
            
            # Create tasks for selected agents
            tasks = await self._create_tasks_for_request(request, agent_selection)
            
            # Execute tasks
            results = await self._execute_agent_tasks(tasks)
            
            # Combine results
            combined_result = await self._combine_results(results, request)
            
            # Log the request and response
            await self._log_user_request(request, combined_result)
            
            # Send Telegram notification for chat interaction
            if request_type == "chat" or request_type == "general":
                agents_used = [agent.get("agent_name", "Unknown Agent") for agent in combined_result.get("agents_used", [])]
                processing_time = combined_result.get("processing_time", 0)
                
                await telegram_service.notify_chat_message(
                    user_message=user_message,
                    agent_response=combined_result.get("summary", "No response generated"),
                    agents_used=agents_used,
                    processing_time=processing_time
                )
            
            return combined_result
            
        except Exception as e:
            log_error(f"Failed to process user request: {e}")
            return {
                "status": "error",
                "error": str(e),
                "request_id": request.get("request_id", "unknown")
            }
    
    async def _analyze_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request to determine which agents should handle it"""
        
        user_message = request.get("message", "").lower()
        request_type = request.get("request_type", "general")
        
        selected_agents = []
        reasoning = []
        
        # Pre-market analysis
        if any(keyword in user_message for keyword in ["premarket", "pre-market", "before market", "early morning"]):
            selected_agents.append("premarket_analyzer_001")
            reasoning.append("Pre-market analysis requested")
        
        # Market analysis
        if any(keyword in user_message for keyword in ["market", "analysis", "trend", "technical", "fundamental"]):
            selected_agents.append("premium_strategy_agent_001")
            reasoning.append("Market analysis requested")
        
        # Portfolio management
        if any(keyword in user_message for keyword in ["portfolio", "holdings", "positions", "balance", "p&l"]):
            selected_agents.append("premium_strategy_agent_001")
            reasoning.append("Portfolio management requested")
        
        # Strategy building
        if any(keyword in user_message for keyword in ["strategy", "build strategy", "create strategy", "trading strategy"]):
            selected_agents.append("premium_strategy_agent_001")
            reasoning.append("Strategy building requested")
        
        # Premium strategy (advanced)
        if any(keyword in user_message for keyword in ["premium", "advanced", "ai strategy", "optimize strategy"]):
            selected_agents.append("premium_strategy_agent_001")
            reasoning.append("Premium strategy requested")
        
        # Risk management
        if any(keyword in user_message for keyword in ["risk", "stop loss", "position size", "drawdown"]):
            selected_agents.append("risk_manager_001")
            reasoning.append("Risk management requested")
        
        # Order execution
        if any(keyword in user_message for keyword in ["buy", "sell", "order", "trade", "execute"]):
            selected_agents.append("order_executor_001")
            reasoning.append("Order execution requested")
        
        # Default to premium strategy agent if no specific agent selected
        if not selected_agents:
            selected_agents.append("premium_strategy_agent_001")
            reasoning.append("Default to premium strategy agent")
        
        return {
            "selected_agents": selected_agents,
            "reasoning": reasoning,
            "confidence": min(len(selected_agents) * 0.2, 1.0)
        }
    
    async def _create_tasks_for_request(self, request: Dict[str, Any], 
                                      agent_selection: Dict[str, Any]) -> List[AgentTask]:
        """Create tasks for selected agents"""
        
        tasks = []
        user_message = request.get("message", "")
        request_context = request.get("context", {})
        
        for agent_id in agent_selection["selected_agents"]:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            
            # Create task based on agent type
            task_data = {
                "user_message": user_message,
                "context": request_context,
                "request_id": request.get("request_id", str(uuid.uuid4())),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add agent-specific task data
            if agent.config.agent_type == "premarket_analyzer":
                task_data["task_type"] = "full_analysis"
            elif agent.config.agent_type == "market_analyzer":
                task_data["task_type"] = "market_analysis"
            elif agent.config.agent_type == "portfolio_manager":
                task_data["task_type"] = "portfolio_analysis"
            elif agent.config.agent_type == "strategy_builder":
                task_data["task_type"] = "build_strategy"
            elif agent.config.agent_type == "premium_strategy_agent":
                task_data["task_type"] = "build_strategy"
            elif agent.config.agent_type == "risk_manager":
                task_data["task_type"] = "risk_assessment"
            elif agent.config.agent_type == "order_executor":
                task_data["task_type"] = "execute_order"
            else:
                task_data["task_type"] = "general_analysis"
            
            task = AgentTask(
                task_id=str(uuid.uuid4()),
                agent_id=agent_id,
                task_type=task_data["task_type"],
                priority=5,
                data=task_data,
                created_at=datetime.now()
            )
            
            tasks.append(task)
        
        return tasks
    
    async def _execute_agent_tasks(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Execute tasks across multiple agents"""
        
        results = []
        
        # Execute tasks concurrently
        task_futures = []
        
        for task in tasks:
            if task.agent_id in self.agents:
                agent = self.agents[task.agent_id]
                future = asyncio.create_task(agent.execute_task(task))
                task_futures.append((task, future))
        
        # Wait for all tasks to complete
        for task, future in task_futures:
            try:
                result = await future
                results.append({
                    "task_id": task.task_id,
                    "agent_id": task.agent_id,
                    "agent_name": self.agents[task.agent_id].config.name,
                    "task_type": task.task_type,
                    "status": "completed",
                    "result": result,
                    "execution_time": task.execution_time
                })
                
                self.tasks_processed += 1
                
            except Exception as e:
                log_error(f"Task {task.task_id} failed: {e}")
                results.append({
                    "task_id": task.task_id,
                    "agent_id": task.agent_id,
                    "agent_name": self.agents[task.agent_id].config.name,
                    "task_type": task.task_type,
                    "status": "failed",
                    "error": str(e)
                })
                
                self.tasks_failed += 1
        
        return results
    
    async def _combine_results(self, results: List[Dict[str, Any]], 
                             original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple agents"""
        
        combined_result = {
            "request_id": original_request.get("request_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "agents_used": [],
            "responses": [],
            "summary": "",
            "recommendations": [],
            "status": "completed"
        }
        
        # Collect agent responses
        for result in results:
            agent_info = {
                "agent_id": result["agent_id"],
                "agent_name": result["agent_name"],
                "task_type": result["task_type"],
                "status": result["status"]
            }
            
            combined_result["agents_used"].append(agent_info)
            
            if result["status"] == "completed":
                response_data = result["result"]
                
                # Extract key information from agent response
                agent_response = {
                    "agent_name": result["agent_name"],
                    "response": response_data,
                    "key_insights": self._extract_key_insights(response_data),
                    "recommendations": self._extract_recommendations(response_data)
                }
                
                combined_result["responses"].append(agent_response)
                
                # Collect recommendations
                recommendations = self._extract_recommendations(response_data)
                combined_result["recommendations"].extend(recommendations)
        
        # Generate summary
        combined_result["summary"] = await self._generate_response_summary(
            original_request.get("message", ""), combined_result["responses"]
        )
        
        return combined_result
    
    def _extract_key_insights(self, response_data: Dict[str, Any]) -> List[str]:
        """Extract key insights from agent response"""
        
        insights = []
        
        # Look for common insight patterns
        if "insights" in response_data:
            if isinstance(response_data["insights"], list):
                insights.extend(response_data["insights"])
            elif isinstance(response_data["insights"], dict):
                insights.append(str(response_data["insights"]))
        
        if "analysis" in response_data:
            insights.append(str(response_data["analysis"]))
        
        if "market_outlook" in response_data:
            insights.append(f"Market outlook: {response_data['market_outlook']}")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _extract_recommendations(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract recommendations from agent response"""
        
        recommendations = []
        
        if "recommendations" in response_data:
            recs = response_data["recommendations"]
            if isinstance(recs, list):
                recommendations.extend(recs)
            elif isinstance(recs, dict):
                recommendations.append(recs)
        
        return recommendations
    
    async def _generate_response_summary(self, user_message: str, 
                                       agent_responses: List[Dict[str, Any]]) -> str:
        """Generate a summary of the response"""
        
        if not agent_responses:
            return "No agents were able to process your request."
        
        # Count agents used
        agent_count = len(agent_responses)
        agent_names = [resp["agent_name"] for resp in agent_responses]
        
        # Generate summary
        summary = f"Your request was processed by {agent_count} agents: {', '.join(agent_names)}. "
        
        # Add key insights
        all_insights = []
        for response in agent_responses:
            all_insights.extend(response.get("key_insights", []))
        
        if all_insights:
            summary += f"Key insights: {', '.join(all_insights[:3])}. "
        
        # Add recommendations count
        total_recommendations = sum(len(resp.get("recommendations", [])) for resp in agent_responses)
        if total_recommendations > 0:
            summary += f"Generated {total_recommendations} recommendations."
        
        return summary
    
    async def _process_tasks(self):
        """Process tasks from the queue"""
        
        while self.running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Process task
                await self._process_orchestrator_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log_error(f"Error processing orchestrator task: {e}")
    
    async def _process_orchestrator_task(self, task: Dict[str, Any]):
        """Process a task from the orchestrator queue"""
        
        task_type = task.get("task_type")
        
        if task_type == "user_request":
            result = await self.process_user_request(task.get("request", {}))
            # Store result or send response
        elif task_type == "agent_coordination":
            await self._coordinate_agents(task.get("coordination_data", {}))
        
    async def _monitor_agents(self):
        """Monitor agent health and performance"""
        
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check agent status
                for agent_id, agent in self.agents.items():
                    status = await agent.get_status()
                    
                    # Log if agent is in error state
                    if status["status"] == "error":
                        log_error(f"Agent {agent_id} is in error state")
                    
                    # Save metrics
                    await self.data_store.save_agent_metric(
                        agent_id, "tasks_completed", status["tasks_completed"]
                    )
                    await self.data_store.save_agent_metric(
                        agent_id, "tasks_failed", status["tasks_failed"]
                    )
                
            except Exception as e:
                log_error(f"Error monitoring agents: {e}")
    
    async def _coordinate_agent_communication(self):
        """Coordinate communication between agents"""
        
        while self.running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check for agent coordination needs
                await self._check_coordination_needs()
                
            except Exception as e:
                log_error(f"Error coordinating agent communication: {e}")
    
    async def _check_coordination_needs(self):
        """Check if agents need coordination"""
        
        # Check if market analyzer has new data that portfolio manager needs
        market_agent = self.agents.get("market_analyzer_001")
        portfolio_agent = self.agents.get("portfolio_manager_001")
        
        if market_agent and portfolio_agent:
            # Trigger coordination if needed
            pass
    
    async def _coordinate_agents(self, coordination_data: Dict[str, Any]):
        """Coordinate between agents"""
        
        coordination_type = coordination_data.get("type")
        
        if coordination_type == "data_sharing":
            await self._share_agent_data(coordination_data)
        elif coordination_type == "task_delegation":
            await self._delegate_tasks(coordination_data)
    
    async def _share_agent_data(self, coordination_data: Dict[str, Any]):
        """Share data between agents"""
        
        from_agent = coordination_data.get("from_agent")
        to_agent = coordination_data.get("to_agent")
        data = coordination_data.get("data")
        
        if from_agent in self.agents and to_agent in self.agents:
            await self.communication_layer.send_message(
                from_agent, to_agent, 
                MessageType.DATA_RESPONSE,
                data
            )
    
    async def _delegate_tasks(self, coordination_data: Dict[str, Any]):
        """Delegate tasks between agents"""
        
        from_agent = coordination_data.get("from_agent")
        to_agent = coordination_data.get("to_agent")
        task_data = coordination_data.get("task_data")
        
        if from_agent in self.agents and to_agent in self.agents:
            await self.communication_layer.send_message(
                from_agent, to_agent,
                MessageType.TASK_REQUEST,
                task_data
            )
    
    async def _log_user_request(self, request: Dict[str, Any], response: Dict[str, Any]):
        """Log user request and response"""
        
        log_data = {
            "request_id": request.get("request_id"),
            "timestamp": datetime.now().isoformat(),
            "request": request,
            "response": response,
            "agents_used": len(response.get("agents_used", [])),
            "processing_time": (datetime.now() - datetime.fromisoformat(request.get("timestamp", datetime.now().isoformat()))).total_seconds()
        }
        
        # Save to data store
        await self.data_store.save_agent_metric(
            "orchestrator", "user_requests_processed", 1
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = await agent.get_status()
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "orchestrator_status": "running" if self.running else "stopped",
            "uptime_seconds": uptime,
            "total_agents": len(self.agents),
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "success_rate": (self.tasks_processed / (self.tasks_processed + self.tasks_failed) * 100) if (self.tasks_processed + self.tasks_failed) > 0 else 0,
            "agent_statuses": agent_statuses,
            "communication_layer_status": await self.communication_layer.get_system_status(),
            "data_store_status": await self.data_store.get_system_overview()
        }
    
    async def schedule_agent_task(self, agent_id: str, task_data: Dict[str, Any], 
                                scheduled_time: datetime = None) -> str:
        """Schedule a task for an agent"""
        
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_id=agent_id,
            task_type=task_data.get("task_type", "scheduled"),
            priority=task_data.get("priority", 5),
            data=task_data,
            created_at=datetime.now(),
            scheduled_at=scheduled_time or datetime.now()
        )
        
        await self.communication_layer.schedule_task(task)
        
        return task.task_id
    
    async def get_agent_list(self) -> List[Dict[str, Any]]:
        """Get list of all agents"""
        
        agents = []
        for agent_id, agent in self.agents.items():
            agents.append({
                "agent_id": agent_id,
                "name": agent.config.name,
                "type": agent.config.agent_type,
                "status": agent.status.value,
                "enabled": agent.config.enabled,
                "capabilities": [cap.value for cap in await agent.get_capabilities()]
            })
        
        return agents
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        
        log_info("Shutting down Agent Orchestrator")
        
        # Stop all agents
        await self.stop_all_agents()
        
        # Shutdown communication layer
        await self.communication_layer.shutdown()
        
        # Close data store
        await self.data_store.close()
        
        self.running = False
        
        log_info("Agent Orchestrator shutdown complete")
