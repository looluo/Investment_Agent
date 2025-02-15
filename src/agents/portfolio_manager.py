from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from tools.openrouter_config import get_chat_completion
import json

from agents.state import AgentState, show_agent_reasoning


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get the technical analyst, fundamentals agent, and risk management agent messages
    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(
        msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent")
    valuation_message = next(
        msg for msg in state["messages"] if msg.name == "valuation_agent")
    risk_message = next(
        msg for msg in state["messages"] if msg.name == "risk_management_agent")

    # Create the system message
    system_message = {
        "role": "system",
        "content": """你是一名投资组合经理，负责根据团队的分析做出最终交易决策。
            你的工作是根据团队的分析做出交易决策，同时严格遵守风险管理约束。

            风险管理约束：
            - 你绝对不能超过风险经理指定的最大持仓规模
            - 你必须遵循风险管理建议的交易操作（买入/卖出/持有）
            - 这些是硬性约束，不能被其他信号覆盖

            在权衡不同信号的方向和时机时：
            1. 估值分析（35%权重）
               - 公允价值评估的主要驱动因素
               - 确定价格是否提供了良好的入场/出场点

            2. 基本面分析（30%权重）
               - 业务质量和增长评估
               - 确定长期潜力的信心

            3. 技术分析（25%权重）
               - 次要确认
               - 帮助确定入场/出场时机

            4. 情绪分析（10%权重）
               - 最后的考虑因素
               - 可以在风险限制内影响仓位大小

            决策过程应为：
            1. 首先检查风险管理约束
            2. 然后评估估值信号
            3. 接着评估基本面信号
            4. 使用技术分析确定时机
            5. 考虑情绪进行最终调整

            在你的输出中提供以下内容：
            - "action": "买入" | "卖出" | "持有",
            - "quantity": <正整数>
            - "confidence": <0到1之间的浮点数>
            - "agent_signals": <包含代理名称、信号（看多 | 看空 | 中性）及其信心的代理信号列表>
            - "reasoning": <决策的简明解释，包括你如何权衡信号>

            交易规则：
            - 绝不超出风险管理仓位限制
            - 只有在有可用现金时才能买入
            - 只有在有股票可卖时才能卖出
            - 卖出数量必须 ≤ 当前持仓
            - 买入数量必须 ≤ 风险管理中的最大持仓规模"""
    }

    # Create the user message
    user_message = {
        "role": "user",
        "content": f"""根据以下团队的分析，做出你的交易决策。

            技术分析交易信号: {technical_message.content}
            基本面分析交易信号: {fundamentals_message.content}
            情绪分析交易信号: {sentiment_message.content}
            估值分析交易信号: {valuation_message.content}
            风险管理交易信号: {risk_message.content}

            当前投资组合如下:
            投资组合:
            现金: {portfolio['cash']:.2f}
            当前持仓: {portfolio['stock']} 股

            你的输出中仅包含action、quantity、reasoning、confidence和agent_signals，并以JSON格式输出。不要包含任何JSON标记。

            记住，action必须是buy、sell或hold。
            只有在有可用现金时才能买入。
            只有在有股票可卖时才能卖出。"""
    }

    # Get the completion from OpenRouter
    result = get_chat_completion([system_message, user_message])

    # 如果API调用失败，使用默认的保守决策
    if result is None:
        result = json.dumps({
            "action": "hold",
            "quantity": 0,
            "confidence": 0.7,
            "agent_signals": [
                {
                    "agent_name": "technical_analysis",
                    "signal": "neutral",
                    "confidence": 0.0
                },
                {
                    "agent_name": "fundamental_analysis",
                    "signal": "bullish",
                    "confidence": 1.0
                },
                {
                    "agent_name": "sentiment_analysis",
                    "signal": "bullish",
                    "confidence": 0.6
                },
                {
                    "agent_name": "valuation_analysis",
                    "signal": "bearish",
                    "confidence": 0.67
                },
                {
                    "agent_name": "risk_management",
                    "signal": "hold",
                    "confidence": 1.0
                }
            ],
            "reasoning": "API调用发生错误。遵循风险管理信号保持持有。这是一个基于混合信号的保守决策：看多的基本面和情绪分析 vs 看空的估值分析，技术分析为中性。"
        })

    # Create the portfolio management message
    message = HumanMessage(
        content=result,
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {
        "messages": state["messages"] + [message],
        "data":state["data"],
        }


def format_decision(action: str, quantity: int, confidence: float, agent_signals: list, reasoning: str) -> dict:
    """Format the trading decision into a standardized output format.
    Think in English but output analysis in Chinese."""

    # 获取各个agent的信号
    fundamental_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "fundamental_analysis"), None)
    valuation_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "valuation_analysis"), None)
    technical_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "technical_analysis"), None)
    sentiment_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "sentiment_analysis"), None)
    risk_signal = next(
        (signal for signal in agent_signals if signal["agent_name"] == "risk_management"), None)

    # 转换信号为中文
    def signal_to_chinese(signal):
        if not signal:
            return "无数据"
        if signal["signal"] == "bullish":
            return "看多"
        elif signal["signal"] == "bearish":
            return "看空"
        return "中性"

    # 创建详细分析报告
    detailed_analysis = f"""
====================================
          投资分析报告
====================================

一、策略分析

1. 基本面分析 (权重30%):
   信号: {signal_to_chinese(fundamental_signal)}
   置信度: {fundamental_signal['confidence']*100:.0f}%
   要点: 
   - 盈利能力: {fundamental_signal.get('reasoning', {}).get('profitability_signal', {}).get('details', '无数据')}
   - 增长情况: {fundamental_signal.get('reasoning', {}).get('growth_signal', {}).get('details', '无数据')}
   - 财务健康: {fundamental_signal.get('reasoning', {}).get('financial_health_signal', {}).get('details', '无数据')}
   - 估值水平: {fundamental_signal.get('reasoning', {}).get('price_ratios_signal', {}).get('details', '无数据')}

2. 估值分析 (权重35%):
   信号: {signal_to_chinese(valuation_signal)}
   置信度: {valuation_signal['confidence']*100:.0f}%
   要点:
   - DCF估值: {valuation_signal.get('reasoning', {}).get('dcf_analysis', {}).get('details', '无数据')}
   - 所有者收益法: {valuation_signal.get('reasoning', {}).get('owner_earnings_analysis', {}).get('details', '无数据')}

3. 技术分析 (权重25%):
   信号: {signal_to_chinese(technical_signal)}
   置信度: {technical_signal['confidence']*100:.0f}%
   要点:
   - 趋势跟踪: ADX={technical_signal.get('strategy_signals', {}).get('trend_following', {}).get('metrics', {}).get('adx', '无数据'):.2f}
   - 均值回归: RSI(14)={technical_signal.get('strategy_signals', {}).get('mean_reversion', {}).get('metrics', {}).get('rsi_14', '无数据'):.2f}
   - 动量指标: 
     * 1月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_1m', '无数据'):.2%}
     * 3月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_3m', '无数据'):.2%}
     * 6月动量={technical_signal.get('strategy_signals', {}).get('momentum', {}).get('metrics', {}).get('momentum_6m', '无数据'):.2%}
   - 波动性: {technical_signal.get('strategy_signals', {}).get('volatility', {}).get('metrics', {}).get('historical_volatility', '无数据'):.2%}

4. 情绪分析 (权重10%):
   信号: {signal_to_chinese(sentiment_signal)}
   置信度: {sentiment_signal['confidence']*100:.0f}%
   分析: {sentiment_signal.get('reasoning', '无详细分析')}

二、风险评估
风险评分: {risk_signal.get('risk_score', '无数据')}/10
主要指标:
- 波动率: {risk_signal.get('risk_metrics', {}).get('volatility', '无数据')*100:.1f}%
- 最大回撤: {risk_signal.get('risk_metrics', {}).get('max_drawdown', '无数据')*100:.1f}%
- VaR(95%): {risk_signal.get('risk_metrics', {}).get('value_at_risk_95', '无数据')*100:.1f}%
- 市场风险: {risk_signal.get('risk_metrics', {}).get('market_risk_score', '无数据')}/10

三、投资建议
操作建议: {'买入' if action == 'buy' else '卖出' if action == 'sell' else '持有'}
交易数量: {quantity}股
决策置信度: {confidence*100:.0f}%

四、决策依据
{reasoning}

===================================="""

    return {
        "action": action,
        "quantity": quantity,
        "confidence": confidence,
        "agent_signals": agent_signals,
        "分析报告": detailed_analysis
    }
