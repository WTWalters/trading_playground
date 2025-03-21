# TITAN Trading System UI/UX Architecture

## 1. Executive Summary

This document outlines the UI/UX architecture for the TITAN Trading System's Django web application. The architecture focuses on delivering a high-performance, data-rich interface optimized for swing trading with statistical arbitrage strategies. Following Edward Tufte's visualization principles and John Carmack's performance optimization techniques, the UI is designed to support critical trading workflows with minimal latency (<100ms) and maximum clarity.

## 2. Core Design Principles

### 2.1 Tufte-Inspired Visualization Principles
- **Maximum Data-Ink Ratio**: Eliminate non-essential UI elements
- **Small Multiples**: Use for quick comparison of trading pairs
- **Layering and Separation**: Clear visual hierarchy for decision flow
- **Micro/Macro Readings**: Support both detailed and overview insights
- **Integration of Evidence**: Combine multiple data types in single views
- **Documentation of Sources**: Show data lineage for trust
- **Content Density**: High information density without visual clutter
- **Avoid Chartjunk**: No decorative elements that distract from data

### 2.2 Performance Principles
- **Latency Targets**: <100ms end-to-end for all updates
- **Rendering Efficiency**: Optimized D3.js visualizations
- **Real-time Updates**: WebSocket-based architecture
- **Resource Optimization**: M3 hardware acceleration

### 2.3 Workflow-Driven Design
- Natural progression through trading decision process
- Focus on critical paths for swing trading (3-5 day holds)
- Support for statistical arbitrage pair selection and execution

## 3. Technology Stack

### 3.1 Frontend Framework
- **React with TypeScript**
- Material UI with custom theming
- React Router for navigation
- React Query for server-state management

### 3.2 Visualization Libraries
- **D3.js** for core trading visualizations
- Lightweight Charts by TradingView for financial charts
- Recharts for statistical visualizations
- Visx for ML visualization components

### 3.3 Backend Integration
- Django REST Framework for API endpoints
- Django Channels for WebSocket support
- JWT authentication with refresh tokens
- Optimized database queries for time-series data

## 4. Core Visualization Components

### 4.1 Candlestick Chart with Entry/Exit Signals
- **Metrics Displayed**: OHLCV, EMA 5/10, z-score signals, entry/exit points, stop-loss (20%)
- **Technical Implementation**: D3.js, sparse design with entry dots at z < -2, exit dots at z > 0
- **User Workflow**: Precise timing of 3-5 day swing trades

### 4.2 Z-Score Small Multiples
- **Metrics Displayed**: Pair z-scores, cointegration p-values (<0.05), spread values
- **Technical Implementation**: Small multiples pattern for pairs (GLD/SLV, SPY/IVV)
- **User Workflow**: Quick comparison of multiple trading pairs

### 4.3 Kelly Range Frame
- **Metrics Displayed**: f* line (0-0.02), 2% position cap, risk/reward visualization
- **Technical Implementation**: Range frame with position size bars
- **User Workflow**: Optimal position sizing with visual risk constraints

### 4.4 Regime Bands
- **Metrics Displayed**: VIX trends, volatility clusters (crisis/stable/bull), macro indicators
- **Technical Implementation**: Time-series bands with color-coded regimes
- **User Workflow**: Strategy adaptation based on market conditions

### 4.5 Risk Strip
- **Metrics Displayed**: 10% market drop probabilities, historical comparison markers (2008, 2020)
- **Technical Implementation**: Horizontal strip chart with probability visualization
- **User Workflow**: Assess tail risk and survival probability

### 4.6 Confidence Tracker
- **Metrics Displayed**: Signal confidence score (0-1), win rate (70% target), journal entries
- **Technical Implementation**: Confidence line visualization with journal integration
- **User Workflow**: Build discipline and trading psychology awareness

## 5. Component Architecture

```
/components
  /core
    - Layout.tsx                  # Main application layout
    - Navigation.tsx              # Minimal, task-focused navigation
    - AuthGuard.tsx               # Authentication wrapper
    - ErrorBoundary.tsx           # Error handling for components
  /trading
    - CandlestickChart.tsx        # Livermore-inspired entry/exit visualization
    - ZScoreMultiples.tsx         # Simons-inspired pair comparison
    - TradeSparklines.tsx         # Compact trade history visualization
  /risk
    - KellyRangeFrame.tsx         # Thorp-inspired position sizing
    - RiskStrip.tsx               # Taleb-inspired tail risk assessment
    - RegimeBands.tsx             # Jones-inspired market regime detection
  /psychology
    - ConfidenceTracker.tsx       # Steenbarger-inspired discipline tool
    - WinRateGauge.tsx            # Duke-inspired probability tracking
    - JournalLog.tsx              # Trading reflection tool
  /common
    - Button.tsx                  # Minimal, purpose-focused buttons
    - Card.tsx                    # Data-dense content cards
    - DataTooltip.tsx             # Context-rich tooltips
```

## 6. Navigation and Information Architecture

### 6.1 Primary Navigation Flow
- **Pair Selection** → **Signal Evaluation** → **Trade Execution** → **Risk Management** → **Performance Reflection**

### 6.2 Interface Organization
- **Pairs Dashboard**: Z-Score Multiples, cointegration metrics, filtering controls
- **Trading View**: Candlestick Chart with entry/exit signals, EMA overlays
- **Position Management**: Kelly Range Frame, current positions, risk metrics
- **Market Context**: Regime Bands, Risk Strip, macro indicators
- **Journal & Analytics**: Confidence Tracker, win rate metrics, reflection tools

## 7. Real-time Data Architecture

### 7.1 WebSocket Integration
- Django Channels for server-side WebSocket support
- Custom React hooks for WebSocket connection management
- Message buffering for connection interruptions
- Optimized binary message format (MessagePack)

### 7.2 WebSocket Channels
```
trading_data
  pair_update                # Real-time pair data updates
  signal_update              # New trading signals
  regime_update              # Market regime changes
  position_update            # Position changes
```

### 7.3 Performance Optimizations
- Selective rendering for UI updates
- Client-side data transformation for visualization efficiency
- Throttling for high-frequency data
- Data caching strategy for frequently accessed information

## 8. API Integration

### 8.1 API Structure
```
/api/v1/
  /pairs/                    # Trading pair data
    GET /                    # List all pairs
    GET /<id>/               # Get specific pair
    GET /<id>/data/          # Get OHLCV data
    GET /<id>/signals/       # Get signals
  
  /analysis/
    GET /regimes/            # Get regime data
    GET /cointegration/      # Get cointegration results
    
  /trading/
    GET /positions/          # Current positions
    POST /orders/            # Submit orders
    GET /performance/        # Performance metrics
    
  /risk/
    GET /kelly/              # Kelly criterion data
    GET /stress/             # Stress test results
```

### 8.2 Data Flow Integration
- TimescaleDB → Django API → WebSocket → React Components
- Market Analysis Components → WebSocket → UI Visualizations
- Strategy Management → WebSocket → Position Visualizations

## 9. Responsive Design Strategy

### 9.1 Device Strategy
- Primary focus on desktop experience (main trading platform)
- Tablet support for monitoring and essential trading
- Mobile support for notifications and basic monitoring

### 9.2 Responsive Approach
- Component-based adaptivity
- Progressive enhancement
- Performance optimization for different devices

### 9.3 Key Breakpoints
- Large Desktop (1920px+): Full trading interface with multiple charts
- Desktop (1366px+): Standard trading interface with side panels
- Tablet (768px-1365px): Simplified interface with prioritized components
- Mobile (320px-767px): Monitoring focus with limited trading capability

## 10. Implementation Plan

### 10.1 Phase 1: Foundation (2 weeks)
- Set up Django project with REST Framework and Channels
- Create React application with TypeScript
- Implement authentication system
- Establish core layout and navigation
- Create basic dashboard page

### 10.2 Phase 2: Data Visualization (3 weeks)
- Implement financial chart components
- Create statistical visualization components
- Develop WebSocket integration for real-time updates
- Build performance chart components
- Create responsive design system

### 10.3 Phase 3: Trading Interface (3 weeks)
- Develop pair analysis components
- Create trading interface with order entry
- Implement position management
- Build settings and configuration pages
- Create user preference management

### 10.4 Phase 4: Advanced Features (2 weeks)
- Implement ML visualization components
- Create advanced analytics dashboards
- Develop notification system
- Build user onboarding experience
- Add final performance optimizations

## 11. Accessibility Considerations

- High contrast mode for trading interfaces
- Keyboard navigation for critical trading functions
- Screen reader compatibility for data visualizations
- Color schemes safe for color-blind users
- Configurable font sizes for readability

## 12. Integration with Existing Architecture

The UI architecture integrates with the existing TITAN Trading System components:

- **Data Infrastructure**: UI connects to TimescaleDB via Django API
- **Market Analysis**: Visualization of cointegration, mean reversion, and regime detection
- **Strategy Management**: Interface for strategy parameters and optimization
- **Execution Framework**: Trade execution and position tracking
- **Risk Management**: Visualization of Kelly criterion, position sizing, and risk controls

## Appendix: Component Examples

### A.1 Candlestick Chart Component
```jsx
// Sparse, high-performance candlestick chart with entry/exit signals
const CandlestickChart = ({ data, signals, ema }) => {
  const svgRef = useRef(null);
  
  useEffect(() => {
    if (!data || data.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;
    
    // Clear previous rendering
    svg.selectAll("*").remove();
    
    // Create scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => d.date))
      .range([40, width - 10]);
      
    const yScale = d3.scaleLinear()
      .domain([
        d3.min(data, d => d.low) * 0.99,
        d3.max(data, d => d.high) * 1.01
      ])
      .range([height - 40, 10]);
    
    // Render candlesticks with minimal ink
    svg.selectAll("line.candle")
      .data(data)
      .enter()
      .append("line")
      .attr("class", "candle")
      .attr("x1", d => xScale(d.date))
      .attr("x2", d => xScale(d.date))
      .attr("y1", d => yScale(d.low))
      .attr("y2", d => yScale(d.high))
      .attr("stroke", d => d.close > d.open ? "#2E8B57" : "#DC143C")
      .attr("stroke-width", 1);
      
    svg.selectAll("rect.candle")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "candle")
      .attr("x", d => xScale(d.date) - 3)
      .attr("y", d => yScale(Math.max(d.open, d.close)))
      .attr("width", 6)
      .attr("height", d => Math.abs(yScale(d.open) - yScale(d.close)))
      .attr("fill", d => d.close > d.open ? "#2E8B57" : "#DC143C");
    
    // Render EMAs with minimal line weight
    if (ema?.ema5) {
      const ema5Line = d3.line()
        .x(d => xScale(d.date))
        .y(d => yScale(d.ema5));
        
      svg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#1E90FF")
        .attr("stroke-width", 1)
        .attr("d", ema5Line);
    }
    
    if (ema?.ema10) {
      const ema10Line = d3.line()
        .x(d => xScale(d.date))
        .y(d => yScale(d.ema10));
        
      svg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#9932CC")
        .attr("stroke-width", 1)
        .attr("d", ema10Line);
    }
    
    // Entry/exit signals with high visual clarity
    svg.selectAll("circle.entry")
      .data(signals.filter(s => s.type === 'entry'))
      .enter()
      .append("circle")
      .attr("class", "entry")
      .attr("cx", d => xScale(d.date))
      .attr("cy", d => yScale(d.price))
      .attr("r", 4)
      .attr("fill", "#2E8B57");
      
    svg.selectAll("circle.exit")
      .data(signals.filter(s => s.type === 'exit'))
      .enter()
      .append("circle")
      .attr("class", "exit")
      .attr("cx", d => xScale(d.date))
      .attr("cy", d => yScale(d.price))
      .attr("r", 4)
      .attr("fill", "#DC143C");
      
    // Stop-loss lines with clear visual priority
    svg.selectAll("line.stop")
      .data(signals.filter(s => s.type === 'entry'))
      .enter()
      .append("line")
      .attr("class", "stop")
      .attr("x1", d => xScale(d.date))
      .attr("x2", width - 10)
      .attr("y1", d => yScale(d.price * 0.8)) // 20% stop-loss
      .attr("y2", d => yScale(d.price * 0.8))
      .attr("stroke", "#FF6347")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "5,5");
  }, [data, signals, ema]);
  
  return (
    <svg ref={svgRef} width="100%" height="400" className="candlestick-chart">
      {/* Rendered by D3 */}
    </svg>
  );
};
```

### A.2 Django Channels WebSocket Consumer
```python
# channels/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from trading.models import TradingPair, Signal

class TradingDataConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        
        # Authentication check
        if not self.user.is_authenticated:
            await self.close()
            return
            
        # Join trading data group
        await self.channel_layer.group_add(
            "trading_data",
            self.channel_name
        )
        
        await self.accept()
        
    async def disconnect(self, close_code):
        # Leave trading data group
        await self.channel_layer.group_discard(
            "trading_data",
            self.channel_name
        )
        
    # Receive message from WebSocket
    async def receive(self, text_data):
        data = json.loads(text_data)
        message_type = data.get("type")
        
        if message_type == "subscribe_pairs":
            pair_ids = data.get("pair_ids", [])
            await self.subscribe_to_pairs(pair_ids)
        
    # Handler for pair updates
    async def trading_pair_update(self, event):
        # Send trading pair update to WebSocket
        await self.send(text_data=json.dumps({
            "type": "pair_update",
            "pair": event["pair"],
            "price_data": event["price_data"],
            "z_score": event["z_score"],
            "signals": event["signals"]
        }))
        
    # Helper to subscribe to specific pairs
    async def subscribe_to_pairs(self, pair_ids):
        # Implementation to handle pair subscriptions
        pass
```
