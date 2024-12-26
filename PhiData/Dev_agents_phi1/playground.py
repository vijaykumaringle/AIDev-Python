from phi.agent import Agent
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.model.groq import Groq
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools as YFinance
from phi.tools.duckduckgo import DuckDuckGo as Web_Search
from phi.tools.googlesearch import GoogleSearch
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv


load_dotenv()

web_agent = Agent(
    name = "Web Agent",
    #model = Groq(id="llama-3.3-70b-versatile"),
    model = Gemini(id="gemini-1.5-flash"),
    #tools=[(search=True, news=True)],
    tools=[GoogleSearch()],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    show_tool_calls=True,
    markdown=True,
    instructions=["Web search for the news realated to the companies and pick first 2 news."]
)

gemini_agent = Agent(
    name = "Gemini Agent",
    model = Gemini(id="gemini-1.5-flash"),
    markdown=True,
    storage=SqlAgentStorage(table_name="gemini_agent", db_file="agents.db"),
    instructions=["Consider you are a poet and writes finance poetries."]
)

finance_agent = Agent(
    name = "Finance Agent",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinance(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    markdown=True,
    instructions=["Use tables to display data."]
)

agent_team = Agent(
    team= [web_agent, finance_agent, gemini_agent],
    show_tool_calls=True,
    markdown=True,
    #model = Groq(id="llama-3.3-70b-versatile"),
    model = Gemini(id="gemini-1.5-flash"),
    instructions=["Use tables to display data.", "Display latest news about the stock.", "Create summary news and recommendation table.", "Write short poem on the companies."]
)


app = Playground(agents=[finance_agent, web_agent, gemini_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
