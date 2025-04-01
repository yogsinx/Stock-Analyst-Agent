from phi.agent import Agent
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAgentPlayground:
    """Playground setup for stock analysis agents."""
    
    def __init__(self):
        """Initialize the playground environment."""
        self._load_environment()
        self.model = Groq(id="llama-3.3-70b-versatile")
        
    def _load_environment(self) -> None:
        """Load environment variables and API keys."""
        try:
            load_dotenv()
            api_key = os.getenv("PHI_API_KEY")
            if not api_key:
                raise ValueError("PHI_API_KEY not found in environment")
            import phi
            phi.api = api_key
        except Exception as e:
            logger.error(f"Environment setup failed: {str(e)}")
            raise

    def create_web_search_agent(self) -> Agent:
        """Create and configure the web search agent."""
        return Agent(
            name="Web search agent",
            role="A web search agent that can find information about stocks.",
            model=self.model,
            instruction=["Always include sources and verify information accuracy."],
            tools=[DuckDuckGo()],
            show_tools_calls=True,
            markdown=True,
        )

    def create_finance_agent(self) -> Agent:
        """Create and configure the finance agent."""
        return Agent(
            name="Finance agent",
            role="A finance agent that provides detailed stock analysis.",
            model=self.model,
            instruction=["Present data in tables with clear explanations."],
            tools=[
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    stock_fundamentals=True,
                    company_info=True,
                    technical_indicators=True
                ),
            ],
            show_tools_calls=True,
            markdown=True,
        )

    def setup_playground(self) -> Playground:
        """Set up and return the playground with configured agents."""
        try:
            agents = [
                self.create_finance_agent(),
                self.create_web_search_agent()
            ]
            return Playground(agents=agents)
        except Exception as e:
            logger.error(f"Playground setup failed: {str(e)}")
            raise

playground = StockAgentPlayground()
app = playground.setup_playground().get_app()

def main():
    """Main function to run the playground."""
    try:
        logger.info("Starting playground server...")
        serve_playground_app("playground:app", reload=True)
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()