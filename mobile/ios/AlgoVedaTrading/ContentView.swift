//
//  ContentView.swift
//  AlgoVeda Trading
//  Professional Mobile Trading Interface for iOS
//

import SwiftUI
import Charts
import Combine
import Network

// MARK: - Data Models

struct PortfolioSummary: Codable, Identifiable {
    let id = UUID()
    let totalValue: Double
    let totalCash: Double
    let totalPnL: Double
    let dailyPnL: Double
    let leverage: Double
    let marginUsed: Double
    let availableMargin: Double
    let numberOfPositions: Int
    let lastUpdated: Date
    
    enum CodingKeys: String, CodingKey {
        case totalValue, totalCash, totalPnL, dailyPnL, leverage
        case marginUsed, availableMargin, numberOfPositions, lastUpdated
    }
}

struct Position: Codable, Identifiable {
    let id = UUID()
    let symbol: String
    let quantity: Double
    let marketValue: Double
    let unrealizedPnL: Double
    let dailyChange: Double
    let dailyChangePercent: Double
    let sector: String
    let averageCost: Double
    let currentPrice: Double
    let weight: Double
    let assetClass: String
    
    enum CodingKeys: String, CodingKey {
        case symbol, quantity, marketValue, unrealizedPnL, dailyChange
        case dailyChangePercent, sector, averageCost, currentPrice, weight, assetClass
    }
}

struct Order: Codable, Identifiable {
    let id = UUID()
    let orderId: String
    let symbol: String
    let side: OrderSide
    let quantity: Double
    let orderType: String
    let status: OrderStatus
    let filledQuantity: Double
    let averagePrice: Double
    let createdAt: Date
    let strategy: String?
    
    enum CodingKeys: String, CodingKey {
        case orderId, symbol, side, quantity, orderType, status
        case filledQuantity, averagePrice, createdAt, strategy
    }
}

enum OrderSide: String, Codable, CaseIterable {
    case buy = "BUY"
    case sell = "SELL"
    
    var color: Color {
        switch self {
        case .buy: return .green
        case .sell: return .red
        }
    }
}

enum OrderStatus: String, Codable {
    case pending = "PENDING"
    case filled = "FILLED"
    case cancelled = "CANCELLED"
    case rejected = "REJECTED"
    case partiallyFilled = "PARTIALLY_FILLED"
    
    var color: Color {
        switch self {
        case .pending: return .orange
        case .filled: return .green
        case .cancelled, .rejected: return .red
        case .partiallyFilled: return .blue
        }
    }
}

struct Trade: Codable, Identifiable {
    let id = UUID()
    let tradeId: String
    let symbol: String
    let side: OrderSide
    let quantity: Double
    let price: Double
    let timestamp: Date
    let commission: Double
    let venue: String
    
    enum CodingKeys: String, CodingKey {
        case tradeId, symbol, side, quantity, price, timestamp, commission, venue
    }
}

struct Alert: Codable, Identifiable {
    let id: String
    let type: AlertType
    let message: String
    let timestamp: Date
    let acknowledged: Bool
    let symbol: String?
    let value: Double?
    let threshold: Double?
}

enum AlertType: String, Codable {
    case info = "INFO"
    case warning = "WARNING"
    case error = "ERROR"
    case success = "SUCCESS"
    
    var color: Color {
        switch self {
        case .info: return .blue
        case .warning: return .orange
        case .error: return .red
        case .success: return .green
        }
    }
    
    var icon: String {
        switch self {
        case .info: return "info.circle"
        case .warning: return "exclamationmark.triangle"
        case .error: return "xmark.circle"
        case .success: return "checkmark.circle"
        }
    }
}

struct MarketData: Codable, Identifiable {
    let id = UUID()
    let symbol: String
    let price: Double
    let change: Double
    let changePercent: Double
    let volume: Double
    let bid: Double
    let ask: Double
    let high: Double
    let low: Double
    let timestamp: Date
    
    enum CodingKeys: String, CodingKey {
        case symbol, price, change, changePercent, volume
        case bid, ask, high, low, timestamp
    }
}

struct TradingStrategy: Codable, Identifiable {
    let id = UUID()
    let strategyId: String
    let name: String
    let status: StrategyStatus
    let pnl: Double
    let sharpeRatio: Double
    let maxDrawdown: Double
    let ordersToday: Int
    let description: String
    
    enum CodingKeys: String, CodingKey {
        case strategyId, name, status, pnl, sharpeRatio
        case maxDrawdown, ordersToday, description
    }
}

enum StrategyStatus: String, Codable {
    case active = "ACTIVE"
    case paused = "PAUSED"
    case stopped = "STOPPED"
    
    var color: Color {
        switch self {
        case .active: return .green
        case .paused: return .orange
        case .stopped: return .red
        }
    }
    
    var icon: String {
        switch self {
        case .active: return "play.circle.fill"
        case .paused: return "pause.circle.fill"
        case .stopped: return "stop.circle.fill"
        }
    }
}

// MARK: - WebSocket Manager

class WebSocketManager: ObservableObject {
    private var webSocketTask: URLSessionWebSocketTask?
    private let urlSession = URLSession.shared
    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var lastMessage: WebSocketMessage?
    
    enum ConnectionStatus {
        case connecting
        case connected
        case disconnected
        case error(String)
        
        var displayName: String {
            switch self {
            case .connecting: return "Connecting"
            case .connected: return "Connected"
            case .disconnected: return "Disconnected"
            case .error(let message): return "Error: \(message)"
            }
        }
        
        var color: Color {
            switch self {
            case .connecting: return .orange
            case .connected: return .green
            case .disconnected: return .gray
            case .error: return .red
            }
        }
    }
    
    func connect() {
        guard let url = URL(string: "wss://api.algoveda.com/ws") else {
            connectionStatus = .error("Invalid URL")
            return
        }
        
        connectionStatus = .connecting
        webSocketTask = urlSession.webSocketTask(with: url)
        webSocketTask?.resume()
        
        receiveMessage()
        
        // Send ping periodically to keep connection alive
        Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { _ in
            self.ping()
        }
        
        connectionStatus = .connected
    }
    
    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        connectionStatus = .disconnected
    }
    
    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    self?.handleTextMessage(text)
                case .data(let data):
                    self?.handleDataMessage(data)
                @unknown default:
                    break
                }
                
                // Continue receiving messages
                DispatchQueue.main.async {
                    self?.receiveMessage()
                }
                
            case .failure(let error):
                DispatchQueue.main.async {
                    self?.connectionStatus = .error(error.localizedDescription)
                }
            }
        }
    }
    
    private func handleTextMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let message = try? JSONDecoder().decode(WebSocketMessage.self, from: data) else {
            return
        }
        
        DispatchQueue.main.async {
            self.lastMessage = message
        }
    }
    
    private func handleDataMessage(_ data: Data) {
        guard let message = try? JSONDecoder().decode(WebSocketMessage.self, from: data) else {
            return
        }
        
        DispatchQueue.main.async {
            self.lastMessage = message
        }
    }
    
    private func ping() {
        webSocketTask?.sendPing { error in
            if let error = error {
                DispatchQueue.main.async {
                    self.connectionStatus = .error(error.localizedDescription)
                }
            }
        }
    }
}

struct WebSocketMessage: Codable {
    let type: String
    let data: Data
    let timestamp: Date
    
    private enum CodingKeys: String, CodingKey {
        case type, data, timestamp
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        type = try container.decode(String.self, forKey: .type)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        
        // Handle different data types based on message type
        let dataContainer = try container.decode(Data.self, forKey: .data)
        data = dataContainer
    }
}

// MARK: - API Manager

class APIManager: ObservableObject {
    private let baseURL = "https://api.algoveda.com/api/v1"
    private var cancellables = Set<AnyCancellable>()
    
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    func fetchPortfolioSummary() -> AnyPublisher<PortfolioSummary, Error> {
        guard let url = URL(string: "\(baseURL)/portfolio/summary") else {
            return Fail(error: URLError(.badURL))
                .eraseToAnyPublisher()
        }
        
        return URLSession.shared.dataTaskPublisher(for: url)
            .map(\.data)
            .decode(type: PortfolioSummary.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func fetchPositions() -> AnyPublisher<[Position], Error> {
        guard let url = URL(string: "\(baseURL)/portfolio/positions") else {
            return Fail(error: URLError(.badURL))
                .eraseToAnyPublisher()
        }
        
        return URLSession.shared.dataTaskPublisher(for: url)
            .map(\.data)
            .decode(type: [Position].self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func fetchOrders() -> AnyPublisher<[Order], Error> {
        guard let url = URL(string: "\(baseURL)/orders") else {
            return Fail(error: URLError(.badURL))
                .eraseToAnyPublisher()
        }
        
        return URLSession.shared.dataTaskPublisher(for: url)
            .map(\.data)
            .decode(type: [Order].self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
    
    func submitOrder(symbol: String, side: OrderSide, quantity: Double, orderType: String, price: Double?) -> AnyPublisher<String, Error> {
        guard let url = URL(string: "\(baseURL)/orders") else {
            return Fail(error: URLError(.badURL))
                .eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let orderRequest = [
            "symbol": symbol,
            "side": side.rawValue,
            "quantity": quantity,
            "orderType": orderType,
            "price": price as Any
        ] as [String: Any]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: orderRequest)
        } catch {
            return Fail(error: error)
                .eraseToAnyPublisher()
        }
        
        return URLSession.shared.dataTaskPublisher(for: request)
            .map { String(data: $0.data, encoding: .utf8) ?? "Unknown" }
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
}

// MARK: - Main Content View

struct ContentView: View {
    @StateObject private var webSocketManager = WebSocketManager()
    @StateObject private var apiManager = APIManager()
    
    @State private var selectedTab = 0
    @State private var portfolioSummary: PortfolioSummary?
    @State private var positions: [Position] = []
    @State private var orders: [Order] = []
    @State private var trades: [Trade] = []
    @State private var alerts: [Alert] = []
    @State private var strategies: [TradingStrategy] = []
    
    @State private var showingOrderEntry = false
    @State private var showingAlert = false
    @State private var alertMessage = ""
    
    var body: some View {
        NavigationView {
            TabView(selection: $selectedTab) {
                // Portfolio Tab
                PortfolioView(portfolioSummary: portfolioSummary, positions: positions)
                    .tabItem {
                        Image(systemName: "chart.pie.fill")
                        Text("Portfolio")
                    }
                    .tag(0)
                
                // Orders Tab
                OrdersView(orders: orders, onNewOrder: { showingOrderEntry = true })
                    .tabItem {
                        Image(systemName: "list.bullet.rectangle")
                        Text("Orders")
                    }
                    .tag(1)
                
                // Trading Tab
                TradingView(onSubmitOrder: submitOrder)
                    .tabItem {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                        Text("Trading")
                    }
                    .tag(2)
                
                // Strategies Tab
                StrategiesView(strategies: strategies)
                    .tabItem {
                        Image(systemName: "brain.head.profile")
                        Text("Strategies")
                    }
                    .tag(3)
                
                // Settings Tab
                SettingsView()
                    .tabItem {
                        Image(systemName: "gear")
                        Text("Settings")
                    }
                    .tag(4)
            }
            .navigationTitle("AlgoVeda")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    ConnectionStatusView(status: webSocketManager.connectionStatus)
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack {
                        if !alerts.filter({ !$0.acknowledged }).isEmpty {
                            Button(action: { showingAlert = true }) {
                                Image(systemName: "bell.fill")
                                    .foregroundColor(.red)
                                    .overlay(
                                        Badge(count: alerts.filter { !$0.acknowledged }.count)
                                    )
                            }
                        }
                        
                        Button(action: refreshData) {
                            Image(systemName: "arrow.clockwise")
                        }
                        .disabled(apiManager.isLoading)
                    }
                }
            }
        }
        .onAppear {
            webSocketManager.connect()
            loadInitialData()
        }
        .onDisappear {
            webSocketManager.disconnect()
        }
        .sheet(isPresented: $showingOrderEntry) {
            OrderEntryView(onSubmit: submitOrder)
        }
        .alert("Alerts", isPresented: $showingAlert) {
            Button("OK") { }
        } message: {
            VStack {
                ForEach(alerts.filter { !$0.acknowledged }.prefix(3)) { alert in
                    Text(alert.message)
                        .foregroundColor(alert.type.color)
                }
            }
        }
        .onChange(of: webSocketManager.lastMessage) { message in
            handleWebSocketMessage(message)
        }
    }
    
    private func loadInitialData() {
        apiManager.isLoading = true
        
        let publishers = Publishers.Zip3(
            apiManager.fetchPortfolioSummary(),
            apiManager.fetchPositions(),
            apiManager.fetchOrders()
        )
        
        publishers
            .sink(
                receiveCompletion: { completion in
                    apiManager.isLoading = false
                    if case .failure(let error) = completion {
                        apiManager.errorMessage = error.localizedDescription
                    }
                },
                receiveValue: { summary, positions, orders in
                    self.portfolioSummary = summary
                    self.positions = positions
                    self.orders = orders
                }
            )
            .store(in: &apiManager.cancellables)
    }
    
    private func refreshData() {
        loadInitialData()
    }
    
    private func submitOrder(symbol: String, side: OrderSide, quantity: Double, orderType: String, price: Double?) {
        apiManager.submitOrder(symbol: symbol, side: side, quantity: quantity, orderType: orderType, price: price)
            .sink(
                receiveCompletion: { completion in
                    if case .failure(let error) = completion {
                        alertMessage = "Order submission failed: \(error.localizedDescription)"
                        showingAlert = true
                    }
                },
                receiveValue: { orderID in
                    alertMessage = "Order submitted successfully: \(orderID)"
                    showingAlert = true
                    refreshData() // Refresh orders list
                }
            )
            .store(in: &apiManager.cancellables)
        
        showingOrderEntry = false
    }
    
    private func handleWebSocketMessage(_ message: WebSocketMessage?) {
        guard let message = message else { return }
        
        switch message.type {
        case "PORTFOLIO_UPDATE":
            // Handle portfolio updates
            break
        case "POSITION_UPDATE":
            // Handle position updates
            break
        case "ORDER_UPDATE":
            // Handle order updates
            refreshData()
        case "PRICE_UPDATE":
            // Handle price updates
            break
        case "ALERT":
            // Handle new alerts
            break
        default:
            break
        }
    }
}

// MARK: - Supporting Views

struct ConnectionStatusView: View {
    let status: WebSocketManager.ConnectionStatus
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(status.color)
                .frame(width: 8, height: 8)
            Text(status.displayName)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

struct Badge: View {
    let count: Int
    
    var body: some View {
        if count > 0 {
            Text("\(count)")
                .font(.caption2)
                .foregroundColor(.white)
                .padding(4)
                .background(Color.red)
                .clipShape(Circle())
                .offset(x: 8, y: -8)
        }
    }
}

struct PortfolioView: View {
    let portfolioSummary: PortfolioSummary?
    let positions: [Position]
    
    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                if let summary = portfolioSummary {
                    PortfolioSummaryCard(summary: summary)
                }
                
                PositionsListView(positions: positions)
                
                AssetAllocationChart(positions: positions)
            }
            .padding()
        }
        .refreshable {
            // Refresh action
        }
    }
}

struct PortfolioSummaryCard: View {
    let summary: PortfolioSummary
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Portfolio Summary")
                    .font(.headline)
                Spacer()
                Text("Updated: \(summary.lastUpdated, formatter: timeFormatter)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                SummaryMetric(title: "Total Value", 
                            value: summary.totalValue, 
                            format: .currency)
                
                SummaryMetric(title: "Daily P&L", 
                            value: summary.dailyPnL, 
                            format: .currency,
                            color: summary.dailyPnL >= 0 ? .green : .red)
                
                SummaryMetric(title: "Available Cash", 
                            value: summary.totalCash, 
                            format: .currency)
                
                SummaryMetric(title: "Margin Used", 
                            value: summary.marginUsed / (summary.marginUsed + summary.availableMargin), 
                            format: .percent)
            }
        }
        .padding()
        .background(Color(UIColor.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct SummaryMetric: View {
    let title: String
    let value: Double
    let format: MetricFormat
    let color: Color?
    
    init(title: String, value: Double, format: MetricFormat, color: Color? = nil) {
        self.title = title
        self.value = value
        self.format = format
        self.color = color
    }
    
    enum MetricFormat {
        case currency
        case percent
        case number
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(formattedValue)
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(color ?? .primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    
    private var formattedValue: String {
        switch format {
        case .currency:
            return currencyFormatter.string(from: NSNumber(value: value)) ?? "$0.00"
        case .percent:
            return percentFormatter.string(from: NSNumber(value: value)) ?? "0%"
        case .number:
            return numberFormatter.string(from: NSNumber(value: value)) ?? "0"
        }
    }
}

struct PositionsListView: View {
    let positions: [Position]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Positions")
                .font(.headline)
            
            LazyVStack(spacing: 8) {
                ForEach(positions) { position in
                    PositionRow(position: position)
                }
            }
        }
    }
}

struct PositionRow: View {
    let position: Position
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(position.symbol)
                    .font(.headline)
                Text(position.sector)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(currencyFormatter.string(from: NSNumber(value: position.marketValue)) ?? "$0")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                HStack(spacing: 4) {
                    Text(position.dailyChange >= 0 ? "+" : "")
                    Text(percentFormatter.string(from: NSNumber(value: position.dailyChangePercent / 100)) ?? "0%")
                }
                .font(.caption)
                .foregroundColor(position.dailyChange >= 0 ? .green : .red)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color(UIColor.systemBackground))
        .cornerRadius(8)
    }
}

struct AssetAllocationChart: View {
    let positions: [Position]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Asset Allocation")
                .font(.headline)
            
            if !positions.isEmpty {
                Chart(positions) { position in
                    SectorMark(
                        angle: .value("Value", position.marketValue),
                        innerRadius: .ratio(0.618),
                        angularInset: 1.5
                    )
                    .cornerRadius(5)
                    .foregroundStyle(by: .value("Symbol", position.symbol))
                }
                .frame(height: 200)
                .chartLegend(position: .bottom, alignment: .center)
            }
        }
        .padding()
        .background(Color(UIColor.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

struct OrdersView: View {
    let orders: [Order]
    let onNewOrder: () -> Void
    
    var body: some View {
        NavigationView {
            List(orders) { order in
                OrderRow(order: order)
            }
            .navigationTitle("Orders")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("New Order", action: onNewOrder)
                }
            }
        }
    }
}

struct OrderRow: View {
    let order: Order
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(order.symbol)
                    .font(.headline)
                HStack {
                    Text(order.side.rawValue)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(order.side.color.opacity(0.2))
                        .foregroundColor(order.side.color)
                        .cornerRadius(4)
                        .font(.caption)
                    
                    Text(order.orderType)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(numberFormatter.string(from: NSNumber(value: order.quantity)) ?? "0")
                    .font(.subheadline)
                
                Text(order.status.rawValue)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(order.status.color.opacity(0.2))
                    .foregroundColor(order.status.color)
                    .cornerRadius(4)
            }
        }
    }
}

struct TradingView: View {
    let onSubmitOrder: (String, OrderSide, Double, String, Double?) -> Void
    
    @State private var selectedSymbol = "AAPL"
    @State private var selectedSide = OrderSide.buy
    @State private var quantity = ""
    @State private var orderType = "MARKET"
    @State private var limitPrice = ""
    
    let symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "BTC-USD"]
    let orderTypes = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    
    var body: some View {
        Form {
            Section("Order Details") {
                Picker("Symbol", selection: $selectedSymbol) {
                    ForEach(symbols, id: \.self) { symbol in
                        Text(symbol).tag(symbol)
                    }
                }
                
                Picker("Side", selection: $selectedSide) {
                    ForEach(OrderSide.allCases, id: \.self) { side in
                        Text(side.rawValue).tag(side)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                
                TextField("Quantity", text: $quantity)
                    .keyboardType(.decimalPad)
                
                Picker("Order Type", selection: $orderType) {
                    ForEach(orderTypes, id: \.self) { type in
                        Text(type).tag(type)
                    }
                }
                
                if orderType == "LIMIT" || orderType == "STOP" || orderType == "STOP_LIMIT" {
                    TextField("Price", text: $limitPrice)
                        .keyboardType(.decimalPad)
                }
            }
            
            Section {
                Button("Submit Order") {
                    submitOrder()
                }
                .disabled(!isValidOrder)
            }
        }
        .navigationTitle("Trading")
    }
    
    private var isValidOrder: Bool {
        !quantity.isEmpty && 
        (orderType == "MARKET" || !limitPrice.isEmpty) &&
        Double(quantity) != nil &&
        (orderType == "MARKET" || Double(limitPrice) != nil)
    }
    
    private func submitOrder() {
        guard let qty = Double(quantity) else { return }
        let price = orderType == "MARKET" ? nil : Double(limitPrice)
        
        onSubmitOrder(selectedSymbol, selectedSide, qty, orderType, price)
        
        // Reset form
        quantity = ""
        limitPrice = ""
    }
}

struct OrderEntryView: View {
    let onSubmit: (String, OrderSide, Double, String, Double?) -> Void
    @Environment(\.presentationMode) var presentationMode
    
    @State private var symbol = ""
    @State private var side = OrderSide.buy
    @State private var quantity = ""
    @State private var orderType = "MARKET"
    @State private var limitPrice = ""
    
    var body: some View {
        NavigationView {
            Form {
                TextField("Symbol", text: $symbol)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .autocapitalization(.allCharacters)
                
                Picker("Side", selection: $side) {
                    ForEach(OrderSide.allCases, id: \.self) { side in
                        Text(side.rawValue).tag(side)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                
                TextField("Quantity", text: $quantity)
                    .keyboardType(.decimalPad)
                
                Picker("Order Type", selection: $orderType) {
                    Text("Market").tag("MARKET")
                    Text("Limit").tag("LIMIT")
                }
                .pickerStyle(SegmentedPickerStyle())
                
                if orderType == "LIMIT" {
                    TextField("Limit Price", text: $limitPrice)
                        .keyboardType(.decimalPad)
                }
            }
            .navigationTitle("New Order")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Submit") {
                        submitOrder()
                    }
                    .disabled(!isValid)
                }
            }
        }
    }
    
    private var isValid: Bool {
        !symbol.isEmpty && 
        !quantity.isEmpty && 
        (orderType == "MARKET" || !limitPrice.isEmpty)
    }
    
    private func submitOrder() {
        guard let qty = Double(quantity) else { return }
        let price = orderType == "MARKET" ? nil : Double(limitPrice)
        
        onSubmit(symbol.uppercased(), side, qty, orderType, price)
        presentationMode.wrappedValue.dismiss()
    }
}

struct StrategiesView: View {
    let strategies: [TradingStrategy]
    
    var body: some View {
        List(strategies) { strategy in
            StrategyRow(strategy: strategy)
        }
        .navigationTitle("Strategies")
    }
}

struct StrategyRow: View {
    let strategy: TradingStrategy
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(strategy.name)
                    .font(.headline)
                Text(strategy.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                HStack(spacing: 4) {
                    Image(systemName: strategy.status.icon)
                        .foregroundColor(strategy.status.color)
                    Text(strategy.status.rawValue)
                        .font(.caption)
                        .foregroundColor(strategy.status.color)
                }
                
                Text(currencyFormatter.string(from: NSNumber(value: strategy.pnl)) ?? "$0")
                    .font(.subheadline)
                    .foregroundColor(strategy.pnl >= 0 ? .green : .red)
            }
        }
    }
}

struct SettingsView: View {
    @State private var enableNotifications = true
    @State private var enableBiometrics = false
    @State private var tradingMode = "LIVE"
    
    var body: some View {
        Form {
            Section("General") {
                Toggle("Push Notifications", isOn: $enableNotifications)
                Toggle("Biometric Authentication", isOn: $enableBiometrics)
                
                Picker("Trading Mode", selection: $tradingMode) {
                    Text("Paper Trading").tag("PAPER")
                    Text("Live Trading").tag("LIVE")
                }
            }
            
            Section("Account") {
                NavigationLink("Profile") {
                    Text("Profile Settings")
                }
                NavigationLink("Security") {
                    Text("Security Settings")
                }
            }
            
            Section("Support") {
                NavigationLink("Help Center") {
                    Text("Help Center")
                }
                NavigationLink("Contact Support") {
                    Text("Contact Support")
                }
            }
            
            Section {
                Text("Version 2.1.0")
                    .foregroundColor(.secondary)
            }
        }
        .navigationTitle("Settings")
    }
}

// MARK: - Formatters

private let currencyFormatter: NumberFormatter = {
    let formatter = NumberFormatter()
    formatter.numberStyle = .currency
    formatter.minimumFractionDigits = 0
    formatter.maximumFractionDigits = 2
    return formatter
}()

private let percentFormatter: NumberFormatter = {
    let formatter = NumberFormatter()
    formatter.numberStyle = .percent
    formatter.minimumFractionDigits = 1
    formatter.maximumFractionDigits = 2
    return formatter
}()

private let numberFormatter: NumberFormatter = {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    formatter.minimumFractionDigits = 0
    formatter.maximumFractionDigits = 2
    return formatter
}()

private let timeFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.timeStyle = .short
    return formatter
}()

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
