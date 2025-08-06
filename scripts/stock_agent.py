from typing import List
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import json
from pathlib import Path

class StockAnalysisSystem:
    """Main system for stock analysis using multiple AI agents."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the stock analysis system with configuration."""
        self.config = self._load_config(config_path)
        self.model = Groq(id=self.config["model_id"])
        self.agents = self._initialize_agents()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(Path(__file__).parent / config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "model_id": "llama-3.3-70b-versatile",
                "agents": {
                    "web_search": {
                        "instructions": ["Always include sources."]
                    },
                    "finance": {
                        "instructions": ["Use tables to present data."]
                    }
                }
            }

    def _initialize_agents(self) -> dict:
        """Initialize all required agents."""
        agents = {}
        
        # Web Search Agent
        agents["web_search"] = Agent(
            name="Web search agent",
            role="A web search agent that can find information about stocks.",
            model=self.model,
            instruction=self.config["agents"]["web_search"]["instructions"],
            tools=[DuckDuckGo()],
            show_tools_calls=True,
            markdown=True,
        )
        
        # Finance Agent
        agents["finance"] = Agent(
            name="Finance agent",
            role="A finance agent that can find information about stocks.",
            model=self.model,
            instruction=self.config["agents"]["finance"]["instructions"],
            tools=[
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    stock_fundamentals=True,
                    technical_indicators=True
                ),
            ],
            show_tools_calls=True,
            markdown=True,
        )
        
        # Multi Agent
        agents["multi"] = Agent(
            team=[agents["web_search"], agents["finance"]],
            model=self.model,
            instructions=self._combine_instructions(),
            show_tools_calls=True,
            markdown=True,
        )
        
        return agents

    def _combine_instructions(self) -> List[str]:
        """Combine instructions from all agents."""
        instructions = []
        for agent_config in self.config["agents"].values():
            instructions.extend(agent_config["instructions"])
        return list(set(instructions))  # Remove duplicates

    def analyze_stock(self, query: str) -> str:
        """Run stock analysis using the multi-agent system."""
        try:
            response = self.agents["multi"].print_response(query)
            print("\n=== Stock Analysis Results ===")
            print(response)
            return response
        except Exception as e:
            error_msg = f"Error analyzing stock: {str(e)}"
            print(f"\nERROR: {error_msg}")
            return error_msg

def main():
    """Main function to demonstrate usage."""
    print("Initializing Stock Analysis System...")
    stock_system = StockAnalysisSystem()
    
    query = "Summarize the latest news about Adobe and its stock price."
    print(f"\nAnalyzing query: {query}")
    print("-" * 50)
    
    result = stock_system.analyze_stock(query)
    
    if __name__ != "__main__":  # For testing/importing
        return result

if __name__ == "__main__":
    main()