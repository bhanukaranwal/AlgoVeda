/*!
 * Regulatory Reporting Engine
 * Comprehensive compliance and regulatory reporting for institutional trading
 */

use std::{
    collections::{HashMap, BTreeMap, VecDeque},
    sync::{Arc, RwLock, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex},
    time::{interval, timeout},
    fs::File,
    io::AsyncWriteExt,
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration, NaiveDate};
use uuid::Uuid;
use csv::WriterBuilder;

use crate::{
    error::{Result, AlgoVedaError},
    trading::{Order, Fill, OrderSide},
    portfolio::Position,
    order_management::OrderRecord,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryConfig {
    pub jurisdictions: Vec<Jurisdiction>,
    pub reporting_frequency: ReportingFrequency,
    pub enable_mifid2: bool,
    pub enable_dodd_frank: bool,
    pub enable_emir: bool,
    pub enable_cftc: bool,
    pub enable_finra: bool,
    pub enable_best_execution: bool,
    pub output_directory: String,
    pub archive_retention_days: u32,
    pub real_time_reporting: bool,
    pub batch_reporting: bool,
    pub encryption_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Jurisdiction {
    US,
    EU,
    UK,
    APAC,
    Canada,
    Australia,
    Japan,
    Singapore,
    HongKong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportingFrequency {
    RealTime,
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
    OnDemand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryReport {
    pub report_id: String,
    pub report_type: ReportType,
    pub jurisdiction: Jurisdiction,
    pub reporting_date: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub firm_identifier: String,
    pub data: ReportData,
    pub validation_status: ValidationStatus,
    pub submission_status: SubmissionStatus,
    pub created_at: DateTime<Utc>,
    pub submitted_at: Option<DateTime<Utc>>,
    pub acknowledgment_received: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    // MiFID II Reports
    TransactionReporting,
    BestExecution,
    MarketMaking,
    SystematicInternaliser,
    InvestmentFirm,
    
    // Dodd-Frank Reports
    SwapDataRepository,
    VolckerRule,
    LiquidityRisk,
    
    // EMIR Reports
    TradeRepository,
    RiskMitigation,
    
    // CFTC Reports
    LargeTraderReporting,
    CommitmentsOfTraders,
    
    // FINRA Reports
    OrderAuditTrail,
    TradeThroughCompliance,
    
    // Generic Reports
    PositionReporting,
    TradeReporting,
    RiskReporting,
    AuditTrail,
    ComplianceBreaches,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportData {
    MiFID2Transaction(MiFID2TransactionReport),
    BestExecution(BestExecutionReport),
    SwapDataRepository(SwapDataReport),
    OrderAuditTrail(OrderAuditTrailReport),
    PositionReport(PositionReport),
    TradeReport(TradeReport),
    RiskReport(RiskReport),
    ComplianceBreaches(ComplianceBreachReport),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiFID2TransactionReport {
    pub transactions: Vec<MiFID2Transaction>,
    pub total_transactions: u64,
    pub total_volume: f64,
    pub total_notional: f64,
    pub currencies: Vec<String>,
    pub venues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiFID2Transaction {
    pub transaction_reference: String,
    pub trading_date_time: DateTime<Utc>,
    pub trading_capacity: String,  // "DEAL", "MTCH", etc.
    pub venue: String,
    pub instrument_id: String,
    pub instrument_id_type: String,  // "ISIN", "MIC", etc.
    pub buy_sell_indicator: String,  // "BUY", "SELL"
    pub quantity: f64,
    pub price: f64,
    pub price_currency: String,
    pub notional_amount: f64,
    pub notional_currency: String,
    pub venue_of_execution: String,
    pub country_of_branch: String,
    pub investment_decision_within_firm: String,
    pub execution_within_firm: String,
    pub client_identification: String,
    pub client_id_type: String,
    pub investment_decision_person: String,
    pub execution_person: String,
    pub order_transmission: String,
    pub waiver_indicator: Vec<String>,
    pub short_selling_indicator: String,
    pub oti_indicator: String,
    pub commodity_derivative_indicator: String,
    pub securities_financing_transaction_indicator: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestExecutionReport {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub execution_venues: Vec<ExecutionVenueReport>,
    pub overall_statistics: ExecutionStatistics,
    pub currency_pairs: HashMap<String, ExecutionStatistics>,
    pub asset_classes: HashMap<String, ExecutionStatistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionVenueReport {
    pub venue_name: String,
    pub venue_lei: String,
    pub venue_mic: String,
    pub order_flow_percentage: f64,
    pub executed_volume_percentage: f64,
    pub average_price_improvement: f64,
    pub average_spread: f64,
    pub average_speed_of_execution: f64,  // milliseconds
    pub likelihood_of_execution: f64,
    pub average_size_of_execution: f64,
    pub payment_for_order_flow: f64,
    pub rebates_received: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    pub total_orders: u64,
    pub total_executed: u64,
    pub total_volume: f64,
    pub total_notional: f64,
    pub average_execution_time: f64,  // milliseconds
    pub price_improvement_rate: f64,
    pub fill_rate: f64,
    pub average_spread_capture: f64,
    pub market_impact_bps: f64,
    pub implementation_shortfall_bps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapDataReport {
    pub swap_transactions: Vec<SwapTransaction>,
    pub total_notional: f64,
    pub currencies: Vec<String>,
    pub counterparties: Vec<String>,
    pub clearing_status: HashMap<String, u64>,  // cleared vs uncleared
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapTransaction {
    pub uti: String,  // Unique Transaction Identifier
    pub prior_uti: Option<String>,
    pub action_type: String,  // "NEW", "MODIFY", "TERMINATE"
    pub product_classification: String,
    pub asset_class: String,
    pub contract_type: String,
    pub underlying_asset: String,
    pub notional_amount: f64,
    pub notional_currency: String,
    pub effective_date: DateTime<Utc>,
    pub maturity_date: DateTime<Utc>,
    pub counterparty_1_lei: String,
    pub counterparty_2_lei: String,
    pub clearing_obligation: bool,
    pub cleared: bool,
    pub central_counterparty: Option<String>,
    pub trade_repository: String,
    pub collateralization: String,
    pub mark_to_market_value: f64,
    pub mark_to_market_currency: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAuditTrailReport {
    pub cat_reporter_imid: String,
    pub firm_routed_id: String,
    pub orders: Vec<CATOrder>,
    pub total_orders: u64,
    pub reporting_period: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CATOrder {
    pub cat_order_id: String,
    pub order_key_date: NaiveDate,
    pub order_id: String,
    pub orig_order_id: Option<String>,
    pub event_timestamp: DateTime<Utc>,
    pub manual_flag: bool,
    pub electronic_flag: bool,
    pub symbol: String,
    pub side: String,
    pub quantity: u64,
    pub price: Option<f64>,
    pub order_type: String,
    pub time_in_force: String,
    pub firm_routed_id: String,
    pub order_received_timestamp: DateTime<Utc>,
    pub route_timestamp: Option<DateTime<Utc>>,
    pub route_id: Option<String>,
    pub session_id: String,
    pub handling_instructions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionReport {
    pub positions: Vec<RegulatoryPosition>,
    pub total_gross_notional: f64,
    pub total_net_notional: f64,
    pub currencies: Vec<String>,
    pub asset_classes: HashMap<String, f64>,
    pub concentration_metrics: ConcentrationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryPosition {
    pub instrument_id: String,
    pub instrument_type: String,
    pub position_date: NaiveDate,
    pub long_quantity: f64,
    pub short_quantity: f64,
    pub net_quantity: f64,
    pub market_value: f64,
    pub currency: String,
    pub venue: String,
    pub asset_class: String,
    pub country_code: String,
    pub issuer_lei: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationMetrics {
    pub largest_position_percentage: f64,
    pub top_5_positions_percentage: f64,
    pub top_10_positions_percentage: f64,
    pub herfindahl_hirschman_index: f64,
    pub sector_concentration: HashMap<String, f64>,
    pub geographic_concentration: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeReport {
    pub trades: Vec<RegulatoryTrade>,
    pub total_trades: u64,
    pub total_volume: f64,
    pub total_notional: f64,
    pub by_venue: HashMap<String, TradeStatistics>,
    pub by_asset_class: HashMap<String, TradeStatistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryTrade {
    pub trade_id: String,
    pub trade_date: NaiveDate,
    pub trade_time: DateTime<Utc>,
    pub settlement_date: NaiveDate,
    pub instrument_id: String,
    pub instrument_type: String,
    pub side: String,
    pub quantity: f64,
    pub price: f64,
    pub currency: String,
    pub venue: String,
    pub counterparty: String,
    pub trade_type: String,  // "REGULAR", "BLOCK", "CROSS", etc.
    pub clearing_member: Option<String>,
    pub trade_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeStatistics {
    pub count: u64,
    pub volume: f64,
    pub notional: f64,
    pub average_trade_size: f64,
    pub largest_trade: f64,
    pub smallest_trade: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskReport {
    pub var_metrics: VaRMetrics,
    pub stress_test_results: Vec<StressTestResult>,
    pub concentration_risks: ConcentrationRisks,
    pub liquidity_metrics: LiquidityMetrics,
    pub counterparty_exposures: Vec<CounterpartyExposure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRMetrics {
    pub one_day_var_95: f64,
    pub one_day_var_99: f64,
    pub ten_day_var_95: f64,
    pub ten_day_var_99: f64,
    pub expected_shortfall_95: f64,
    pub expected_shortfall_99: f64,
    pub var_breaches_ytd: u32,
    pub backtesting_exceptions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    pub scenario_name: String,
    pub scenario_type: String,  // "HISTORICAL", "HYPOTHETICAL", "MONTE_CARLO"
    pub pnl_impact: f64,
    pub percentage_impact: f64,
    pub worst_position: String,
    pub time_to_liquidate_days: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationRisks {
    pub single_name_limit_breaches: Vec<String>,
    pub sector_limit_breaches: Vec<String>,
    pub geographic_limit_breaches: Vec<String>,
    pub currency_limit_breaches: Vec<String>,
    pub largest_single_exposure_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMetrics {
    pub liquidity_coverage_ratio: f64,
    pub net_stable_funding_ratio: f64,
    pub cash_positions: f64,
    pub unencumbered_assets: f64,
    pub committed_facilities: f64,
    pub days_to_liquidate_portfolio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterpartyExposure {
    pub counterparty_lei: String,
    pub counterparty_name: String,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub collateral_held: f64,
    pub collateral_posted: f64,
    pub credit_rating: String,
    pub probability_of_default: f64,
    pub loss_given_default: f64,
    pub expected_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceBreachReport {
    pub breaches: Vec<ComplianceBreach>,
    pub total_breaches: u64,
    pub by_severity: HashMap<String, u64>,
    pub by_type: HashMap<String, u64>,
    pub resolved_breaches: u64,
    pub outstanding_breaches: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceBreach {
    pub breach_id: String,
    pub breach_type: String,
    pub severity: String,  // "LOW", "MEDIUM", "HIGH", "CRITICAL"
    pub detected_at: DateTime<Utc>,
    pub description: String,
    pub affected_instruments: Vec<String>,
    pub estimated_impact: f64,
    pub resolution_status: String,  // "OPEN", "INVESTIGATING", "RESOLVED", "DISMISSED"
    pub assigned_to: String,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution_notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Pending,
    Valid,
    Invalid(Vec<String>),  // List of validation errors
    Warning(Vec<String>),  // List of warnings
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubmissionStatus {
    NotSubmitted,
    Pending,
    Submitted,
    Acknowledged,
    Rejected(String),
    Resubmitted,
}

pub struct RegulatoryReportingEngine {
    config: RegulatoryConfig,
    
    // Report storage
    reports: Arc<RwLock<HashMap<String, RegulatoryReport>>>,
    pending_submissions: Arc<RwLock<VecDeque<String>>>,
    
    // Data collection
    transaction_buffer: Arc<RwLock<VecDeque<TransactionData>>>,
    order_buffer: Arc<RwLock<VecDeque<OrderData>>>,
    position_buffer: Arc<RwLock<VecDeque<PositionData>>>,
    
    // Event handling
    report_events: broadcast::Sender<ReportEvent>,
    
    // Scheduling
    report_scheduler: Arc<Mutex<ReportScheduler>>,
    
    // Performance metrics
    reports_generated: Arc<AtomicU64>,
    reports_submitted: Arc<AtomicU64>,
    validation_errors: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct TransactionData {
    pub order_id: String,
    pub fill: Fill,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub instrument_data: InstrumentData,
}

#[derive(Debug, Clone)]
struct OrderData {
    pub order_record: OrderRecord,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub routing_data: RoutingData,
}

#[derive(Debug, Clone)]
struct PositionData {
    pub position: Position,
    pub timestamp: DateTime<Utc>,
    pub venue: String,
    pub market_data: MarketData,
}

#[derive(Debug, Clone)]
struct InstrumentData {
    pub isin: Option<String>,
    pub mic: Option<String>,
    pub lei: Option<String>,
    pub asset_class: String,
    pub currency: String,
    pub country_code: String,
}

#[derive(Debug, Clone)]
struct RoutingData {
    pub route_id: String,
    pub routing_decision_time: DateTime<Utc>,
    pub routing_person: String,
    pub execution_person: String,
    pub client_id: String,
}

#[derive(Debug, Clone)]
struct MarketData {
    pub last_price: f64,
    pub bid: f64,
    pub ask: f64,
    pub volume: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportEvent {
    pub event_id: String,
    pub report_id: String,
    pub event_type: ReportEventType,
    pub timestamp: DateTime<Utc>,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportEventType {
    ReportGenerated,
    ReportValidated,
    ReportSubmitted,
    ReportAcknowledged,
    ReportRejected,
    ValidationFailed,
    SubmissionFailed,
}

struct ReportScheduler {
    scheduled_reports: BTreeMap<DateTime<Utc>, ScheduledReport>,
    recurring_schedules: Vec<RecurringSchedule>,
}

#[derive(Debug, Clone)]
struct ScheduledReport {
    report_type: ReportType,
    jurisdiction: Jurisdiction,
    period_start: DateTime<Utc>,
    period_end: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct RecurringSchedule {
    report_type: ReportType,
    jurisdiction: Jurisdiction,
    frequency: ReportingFrequency,
    last_generated: Option<DateTime<Utc>>,
    next_due: DateTime<Utc>,
}

impl RegulatoryReportingEngine {
    pub fn new(config: RegulatoryConfig) -> Self {
        let (report_events, _) = broadcast::channel(1000);
        
        Self {
            config,
            reports: Arc::new(RwLock::new(HashMap::new())),
            pending_submissions: Arc::new(RwLock::new(VecDeque::new())),
            transaction_buffer: Arc::new(RwLock::new(VecDeque::new())),
            order_buffer: Arc::new(RwLock::new(VecDeque::new())),
            position_buffer: Arc::new(RwLock::new(VecDeque::new())),
            report_events,
            report_scheduler: Arc::new(Mutex::new(ReportScheduler::new())),
            reports_generated: Arc::new(AtomicU64::new(0)),
            reports_submitted: Arc::new(AtomicU64::new(0)),
            validation_errors: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start the regulatory reporting engine
    pub async fn start(&self) -> Result<()> {
        // Start periodic report generation
        self.start_report_scheduler().await;
        
        // Start submission processor
        self.start_submission_processor().await;
        
        // Start data buffer cleanup
        self.start_buffer_cleanup().await;
        
        Ok(())
    }

    /// Record a transaction for reporting
    pub async fn record_transaction(&self, order_id: String, fill: Fill, venue: String, instrument_data: InstrumentData) {
        let transaction_data = TransactionData {
            order_id,
            fill,
            timestamp: Utc::now(),
            venue,
            instrument_data,
        };
        
        self.transaction_buffer.write().unwrap().push_back(transaction_data);
        
        // Trigger real-time reporting if enabled
        if self.config.real_time_reporting {
            // Would trigger immediate reporting for certain transactions
        }
    }

    /// Record an order for reporting
    pub async fn record_order(&self, order_record: OrderRecord, venue: String, routing_data: RoutingData) {
        let order_data = OrderData {
            order_record,
            timestamp: Utc::now(),
            venue,
            routing_data,
        };
        
        self.order_buffer.write().unwrap().push_back(order_data);
    }

    /// Record a position for reporting
    pub async fn record_position(&self, position: Position, venue: String, market_data: MarketData) {
        let position_data = PositionData {
            position,
            timestamp: Utc::now(),
            venue,
            market_data,
        };
        
        self.position_buffer.write().unwrap().push_back(position_data);
    }

    /// Generate a specific report
    pub async fn generate_report(
        &self,
        report_type: ReportType,
        jurisdiction: Jurisdiction,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> Result<String> {
        let report_id = Uuid::new_v4().to_string();
        
        let report_data = match report_type {
            ReportType::TransactionReporting => {
                self.generate_transaction_report(period_start, period_end).await?
            }
            ReportType::BestExecution => {
                self.generate_best_execution_report(period_start, period_end).await?
            }
            ReportType::OrderAuditTrail => {
                self.generate_order_audit_trail_report(period_start, period_end).await?
            }
            ReportType::PositionReporting => {
                self.generate_position_report(period_start, period_end).await?
            }
            ReportType::TradeReporting => {
                self.generate_trade_report(period_start, period_end).await?
            }
            ReportType::RiskReporting => {
                self.generate_risk_report(period_start, period_end).await?
            }
            ReportType::ComplianceBreaches => {
                self.generate_compliance_breach_report(period_start, period_end).await?
            }
            _ => {
                return Err(AlgoVedaError::Compliance(format!("Report type {:?} not implemented", report_type)));
            }
        };

        let report = RegulatoryReport {
            report_id: report_id.clone(),
            report_type,
            jurisdiction,
            reporting_date: Utc::now(),
            period_start,
            period_end,
            firm_identifier: "ALGOVEDA001".to_string(), // Would be configurable
            data: report_data,
            validation_status: ValidationStatus::Pending,
            submission_status: SubmissionStatus::NotSubmitted,
            created_at: Utc::now(),
            submitted_at: None,
            acknowledgment_received: None,
        };

        // Validate report
        let validation_status = self.validate_report(&report).await;
        let mut final_report = report;
        final_report.validation_status = validation_status;

        // Store report
        self.reports.write().unwrap().insert(report_id.clone(), final_report);

        // Emit event
        let _ = self.report_events.send(ReportEvent {
            event_id: Uuid::new_v4().to_string(),
            report_id: report_id.clone(),
            event_type: ReportEventType::ReportGenerated,
            timestamp: Utc::now(),
            details: HashMap::new(),
        });

        self.reports_generated.fetch_add(1, Ordering::Relaxed);

        Ok(report_id)
    }

    /// Generate MiFID II transaction report
    async fn generate_transaction_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<ReportData> {
        let transaction_buffer = self.transaction_buffer.read().unwrap();
        
        let relevant_transactions: Vec<&TransactionData> = transaction_buffer
            .iter()
            .filter(|t| t.timestamp >= start && t.timestamp <= end)
            .collect();

        let mut mifid2_transactions = Vec::new();
        let mut total_volume = 0.0;
        let mut total_notional = 0.0;
        let mut currencies = std::collections::HashSet::new();
        let mut venues = std::collections::HashSet::new();

        for transaction_data in relevant_transactions {
            let fill = &transaction_data.fill;
            
            let mifid2_tx = MiFID2Transaction {
                transaction_reference: fill.id.clone(),
                trading_date_time: fill.timestamp,
                trading_capacity: "DEAL".to_string(),
                venue: transaction_data.venue.clone(),
                instrument_id: transaction_data.instrument_data.isin.clone().unwrap_or_else(|| "N/A".to_string()),
                instrument_id_type: "ISIN".to_string(),
                buy_sell_indicator: if matches!(fill.side, Some(crate::trading::OrderSide::Buy)) { "BUY" } else { "SELL" }.to_string(),
                quantity: fill.quantity as f64,
                price: fill.price,
                price_currency: transaction_data.instrument_data.currency.clone(),
                notional_amount: fill.quantity as f64 * fill.price,
                notional_currency: transaction_data.instrument_data.currency.clone(),
                venue_of_execution: transaction_data.venue.clone(),
                country_of_branch: transaction_data.instrument_data.country_code.clone(),
                investment_decision_within_firm: "ALGO001".to_string(),
                execution_within_firm: "EXEC001".to_string(),
                client_identification: "CLIENT001".to_string(),
                client_id_type: "LEI".to_string(),
                investment_decision_person: "PERSON001".to_string(),
                execution_person: "PERSON002".to_string(),
                order_transmission: "N".to_string(),
                waiver_indicator: vec![],
                short_selling_indicator: "N".to_string(),
                oti_indicator: "N".to_string(),
                commodity_derivative_indicator: "N".to_string(),
                securities_financing_transaction_indicator: "N".to_string(),
            };

            total_volume += fill.quantity as f64;
            total_notional += fill.quantity as f64 * fill.price;
            currencies.insert(transaction_data.instrument_data.currency.clone());
            venues.insert(transaction_data.venue.clone());

            mifid2_transactions.push(mifid2_tx);
        }

        Ok(ReportData::MiFID2Transaction(MiFID2TransactionReport {
            transactions: mifid2_transactions,
            total_transactions: relevant_transactions.len() as u64,
            total_volume,
            total_notional,
            currencies: currencies.into_iter().collect(),
            venues: venues.into_iter().collect(),
        }))
    }

    /// Generate best execution report
    async fn generate_best_execution_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<ReportData> {
        let transaction_buffer = self.transaction_buffer.read().unwrap();
        let order_buffer = self.order_buffer.read().unwrap();
        
        // Aggregate execution statistics by venue
        let mut venue_stats: HashMap<String, ExecutionVenueStats> = HashMap::new();
        
        for transaction_data in transaction_buffer.iter() {
            if transaction_data.timestamp >= start && transaction_data.timestamp <= end {
                let stats = venue_stats.entry(transaction_data.venue.clone()).or_insert_with(ExecutionVenueStats::new);
                stats.add_transaction(transaction_data);
            }
        }

        let mut execution_venues = Vec::new();
        for (venue_name, stats) in venue_stats {
            execution_venues.push(ExecutionVenueReport {
                venue_name: venue_name.clone(),
                venue_lei: format!("LEI_{}", venue_name), // Would lookup actual LEI
                venue_mic: venue_name.clone(),
                order_flow_percentage: stats.order_flow_percentage,
                executed_volume_percentage: stats.executed_volume_percentage,
                average_price_improvement: stats.average_price_improvement,
                average_spread: stats.average_spread,
                average_speed_of_execution: stats.average_speed_of_execution,
                likelihood_of_execution: stats.likelihood_of_execution,
                average_size_of_execution: stats.average_size_of_execution,
                payment_for_order_flow: 0.0,
                rebates_received: 0.0,
            });
        }

        let overall_statistics = ExecutionStatistics {
            total_orders: order_buffer.len() as u64,
            total_executed: transaction_buffer.len() as u64,
            total_volume: transaction_buffer.iter().map(|t| t.fill.quantity as f64).sum(),
            total_notional: transaction_buffer.iter().map(|t| t.fill.quantity as f64 * t.fill.price).sum(),
            average_execution_time: 50.0, // Would calculate actual
            price_improvement_rate: 0.65,
            fill_rate: 0.95,
            average_spread_capture: 0.3,
            market_impact_bps: 2.5,
            implementation_shortfall_bps: 5.2,
        };

        Ok(ReportData::BestExecution(BestExecutionReport {
            period_start: start,
            period_end: end,
            execution_venues,
            overall_statistics,
            currency_pairs: HashMap::new(), // Would populate for FX
            asset_classes: HashMap::new(),  // Would populate
        }))
    }

    /// Generate order audit trail report
    async fn generate_order_audit_trail_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<ReportData> {
        let order_buffer = self.order_buffer.read().unwrap();
        
        let mut cat_orders = Vec::new();
        
        for order_data in order_buffer.iter() {
            if order_data.timestamp >= start && order_data.timestamp <= end {
                let cat_order = CATOrder {
                    cat_order_id: format!("CAT_{}", order_data.order_record.order.id),
                    order_key_date: order_data.timestamp.date_naive(),
                    order_id: order_data.order_record.order.id.clone(),
                    orig_order_id: order_data.order_record.parent_order_id.clone(),
                    event_timestamp: order_data.timestamp,
                    manual_flag: false,
                    electronic_flag: true,
                    symbol: order_data.order_record.order.symbol.clone(),
                    side: match order_data.order_record.order.side {
                        OrderSide::Buy => "BUY".to_string(),
                        OrderSide::Sell => "SELL".to_string(),
                    },
                    quantity: order_data.order_record.order.quantity,
                    price: order_data.order_record.order.price,
                    order_type: format!("{:?}", order_data.order_record.order.order_type),
                    time_in_force: format!("{:?}", order_data.order_record.order.time_in_force),
                    firm_routed_id: "ALGOVEDA".to_string(),
                    order_received_timestamp: order_data.order_record.creation_time,
                    route_timestamp: Some(order_data.routing_data.routing_decision_time),
                    route_id: Some(order_data.routing_data.route_id.clone()),
                    session_id: "SESSION001".to_string(),
                    handling_instructions: vec![],
                };
                
                cat_orders.push(cat_order);
            }
        }

        Ok(ReportData::OrderAuditTrail(OrderAuditTrailReport {
            cat_reporter_imid: "ALGOVEDA001".to_string(),
            firm_routed_id: "ALGOVEDA".to_string(),
            orders: cat_orders,
            total_orders: cat_orders.len() as u64,
            reporting_period: format!("{} to {}", start.format("%Y-%m-%d"), end.format("%Y-%m-%d")),
        }))
    }

    /// Generate position report
    async fn generate_position_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<ReportData> {
        let position_buffer = self.position_buffer.read().unwrap();
        
        let mut positions = Vec::new();
        let mut total_gross_notional = 0.0;
        let mut total_net_notional = 0.0;
        let mut currencies = std::collections::HashSet::new();
        let mut asset_classes = HashMap::new();

        // Get latest positions for each instrument
        let mut latest_positions: HashMap<String, &PositionData> = HashMap::new();
        
        for position_data in position_buffer.iter() {
            if position_data.timestamp >= start && position_data.timestamp <= end {
                let key = position_data.position.symbol.clone();
                latest_positions.insert(key, position_data);
            }
        }

        for (_, position_data) in latest_positions {
            let pos = &position_data.position;
            let market_value = pos.quantity as f64 * position_data.market_data.last_price;
            
            let regulatory_position = RegulatoryPosition {
                instrument_id: pos.symbol.clone(),
                instrument_type: "EQUITY".to_string(), // Would determine from instrument data
                position_date: end.date_naive(),
                long_quantity: if pos.quantity > 0 { pos.quantity as f64 } else { 0.0 },
                short_quantity: if pos.quantity < 0 { (-pos.quantity) as f64 } else { 0.0 },
                net_quantity: pos.quantity as f64,
                market_value,
                currency: "USD".to_string(), // Would get from instrument data
                venue: position_data.venue.clone(),
                asset_class: "EQUITY".to_string(),
                country_code: "US".to_string(),
                issuer_lei: None,
            };

            total_gross_notional += market_value.abs();
            total_net_notional += market_value;
            currencies.insert("USD".to_string());
            *asset_classes.entry("EQUITY".to_string()).or_insert(0.0) += market_value;

            positions.push(regulatory_position);
        }

        let concentration_metrics = self.calculate_concentration_metrics(&positions);

        Ok(ReportData::PositionReport(PositionReport {
            positions,
            total_gross_notional,
            total_net_notional,
            currencies: currencies.into_iter().collect(),
            asset_classes,
            concentration_metrics,
        }))
    }

    /// Generate trade report
    async fn generate_trade_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<ReportData> {
        let transaction_buffer = self.transaction_buffer.read().unwrap();
        
        let mut trades = Vec::new();
        let mut total_volume = 0.0;
        let mut total_notional = 0.0;
        let mut by_venue: HashMap<String, TradeStatistics> = HashMap::new();
        let mut by_asset_class: HashMap<String, TradeStatistics> = HashMap::new();

        for transaction_data in transaction_buffer.iter() {
            if transaction_data.timestamp >= start && transaction_data.timestamp <= end {
                let fill = &transaction_data.fill;
                let notional = fill.quantity as f64 * fill.price;
                
                let regulatory_trade = RegulatoryTrade {
                    trade_id: fill.id.clone(),
                    trade_date: fill.timestamp.date_naive(),
                    trade_time: fill.timestamp,
                    settlement_date: fill.timestamp.date_naive() + chrono::Duration::days(2), // T+2
                    instrument_id: transaction_data.order_id.clone(), // Would use proper instrument ID
                    instrument_type: "EQUITY".to_string(),
                    side: match fill.side {
                        Some(OrderSide::Buy) => "BUY".to_string(),
                        Some(OrderSide::Sell) => "SELL".to_string(),
                        None => "UNKNOWN".to_string(),
                    },
                    quantity: fill.quantity as f64,
                    price: fill.price,
                    currency: transaction_data.instrument_data.currency.clone(),
                    venue: transaction_data.venue.clone(),
                    counterparty: "EXCHANGE".to_string(),
                    trade_type: "REGULAR".to_string(),
                    clearing_member: None,
                    trade_flags: vec![],
                };

                total_volume += fill.quantity as f64;
                total_notional += notional;

                // Update venue statistics
                let venue_stats = by_venue.entry(transaction_data.venue.clone()).or_insert_with(|| TradeStatistics {
                    count: 0,
                    volume: 0.0,
                    notional: 0.0,
                    average_trade_size: 0.0,
                    largest_trade: 0.0,
                    smallest_trade: f64::MAX,
                });
                
                venue_stats.count += 1;
                venue_stats.volume += fill.quantity as f64;
                venue_stats.notional += notional;
                venue_stats.largest_trade = venue_stats.largest_trade.max(fill.quantity as f64);
                venue_stats.smallest_trade = venue_stats.smallest_trade.min(fill.quantity as f64);
                venue_stats.average_trade_size = venue_stats.volume / venue_stats.count as f64;

                // Update asset class statistics
                let asset_stats = by_asset_class.entry("EQUITY".to_string()).or_insert_with(|| TradeStatistics {
                    count: 0,
                    volume: 0.0,
                    notional: 0.0,
                    average_trade_size: 0.0,
                    largest_trade: 0.0,
                    smallest_trade: f64::MAX,
                });
                
                asset_stats.count += 1;
                asset_stats.volume += fill.quantity as f64;
                asset_stats.notional += notional;
                asset_stats.largest_trade = asset_stats.largest_trade.max(fill.quantity as f64);
                asset_stats.smallest_trade = asset_stats.smallest_trade.min(fill.quantity as f64);
                asset_stats.average_trade_size = asset_stats.volume / asset_stats.count as f64;

                trades.push(regulatory_trade);
            }
        }

        Ok(ReportData::TradeReport(TradeReport {
            trades,
            total_trades: trades.len() as u64,
            total_volume,
            total_notional,
            by_venue,
            by_asset_class,
        }))
    }

    /// Generate risk report
    async fn generate_risk_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<ReportData> {
        // This would integrate with the VaR calculator and risk engine
        let var_metrics = VaRMetrics {
            one_day_var_95: 100000.0,    // Example values
            one_day_var_99: 150000.0,
            ten_day_var_95: 316227.0,
            ten_day_var_99: 474341.0,
            expected_shortfall_95: 120000.0,
            expected_shortfall_99: 200000.0,
            var_breaches_ytd: 3,
            backtesting_exceptions: 1,
        };

        let stress_test_results = vec![
            StressTestResult {
                scenario_name: "2008 Financial Crisis".to_string(),
                scenario_type: "HISTORICAL".to_string(),
                pnl_impact: -500000.0,
                percentage_impact: -15.2,
                worst_position: "AAPL".to_string(),
                time_to_liquidate_days: 3.5,
            },
            StressTestResult {
                scenario_name: "Interest Rate +200bp".to_string(),
                scenario_type: "HYPOTHETICAL".to_string(),
                pnl_impact: -75000.0,
                percentage_impact: -2.3,
                worst_position: "Bond Portfolio".to_string(),
                time_to_liquidate_days: 1.0,
            },
        ];

        let concentration_risks = ConcentrationRisks {
            single_name_limit_breaches: vec![],
            sector_limit_breaches: vec!["Technology".to_string()],
            geographic_limit_breaches: vec![],
            currency_limit_breaches: vec![],
            largest_single_exposure_pct: 8.5,
        };

        let liquidity_metrics = LiquidityMetrics {
            liquidity_coverage_ratio: 1.25,
            net_stable_funding_ratio: 1.15,
            cash_positions: 500000.0,
            unencumbered_assets: 2000000.0,
            committed_facilities: 1000000.0,
            days_to_liquidate_portfolio: 5.2,
        };

        let counterparty_exposures = vec![
            CounterpartyExposure {
                counterparty_lei: "LEI123456789".to_string(),
                counterparty_name: "Major Bank".to_string(),
                gross_exposure: 1000000.0,
                net_exposure: 750000.0,
                collateral_held: 200000.0,
                collateral_posted: 150000.0,
                credit_rating: "AA".to_string(),
                probability_of_default: 0.001,
                loss_given_default: 0.4,
                expected_loss: 300.0,
            },
        ];

        Ok(ReportData::RiskReport(RiskReport {
            var_metrics,
            stress_test_results,
            concentration_risks,
            liquidity_metrics,
            counterparty_exposures,
        }))
    }

    /// Generate compliance breach report
    async fn generate_compliance_breach_report(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<ReportData> {
        // This would integrate with the compliance monitoring system
        let breaches = vec![
            ComplianceBreach {
                breach_id: "BREACH_001".to_string(),
                breach_type: "Position Limit".to_string(),
                severity: "MEDIUM".to_string(),
                detected_at: start + ChronoDuration::hours(2),
                description: "Position in AAPL exceeded 5% limit".to_string(),
                affected_instruments: vec!["AAPL".to_string()],
                estimated_impact: 10000.0,
                resolution_status: "RESOLVED".to_string(),
                assigned_to: "Risk Manager".to_string(),
                resolved_at: Some(start + ChronoDuration::hours(4)),
                resolution_notes: Some("Position reduced to within limits".to_string()),
            },
        ];

        let mut by_severity = HashMap::new();
        let mut by_type = HashMap::new();
        let mut resolved_count = 0;
        let mut outstanding_count = 0;

        for breach in &breaches {
            *by_severity.entry(breach.severity.clone()).or_insert(0) += 1;
            *by_type.entry(breach.breach_type.clone()).or_insert(0) += 1;
            
            if breach.resolution_status == "RESOLVED" {
                resolved_count += 1;
            } else {
                outstanding_count += 1;
            }
        }

        Ok(ReportData::ComplianceBreaches(ComplianceBreachReport {
            total_breaches: breaches.len() as u64,
            by_severity,
            by_type,
            resolved_breaches: resolved_count,
            outstanding_breaches: outstanding_count,
            breaches,
        }))
    }

    /// Validate a report
    async fn validate_report(&self, report: &RegulatoryReport) -> ValidationStatus {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Basic validation
        if report.firm_identifier.is_empty() {
            errors.push("Firm identifier is required".to_string());
        }

        if report.period_end <= report.period_start {
            errors.push("Period end must be after period start".to_string());
        }

        // Jurisdiction-specific validation
        match report.jurisdiction {
            Jurisdiction::EU => {
                if self.config.enable_mifid2 {
                    // MiFID II specific validation
                    if let ReportData::MiFID2Transaction(ref tx_report) = report.data {
                        for tx in &tx_report.transactions {
                            if tx.instrument_id == "N/A" {
                                warnings.push(format!("Missing ISIN for transaction {}", tx.transaction_reference));
                            }
                        }
                    }
                }
            }
            Jurisdiction::US => {
                if self.config.enable_dodd_frank || self.config.enable_finra {
                    // US specific validation
                    if let ReportData::OrderAuditTrail(ref cat_report) = report.data {
                        if cat_report.cat_reporter_imid.is_empty() {
                            errors.push("CAT Reporter IMID is required for US reporting".to_string());
                        }
                    }
                }
            }
            _ => {}
        }

        if !errors.is_empty() {
            self.validation_errors.fetch_add(1, Ordering::Relaxed);
            ValidationStatus::Invalid(errors)
        } else if !warnings.is_empty() {
            ValidationStatus::Warning(warnings)
        } else {
            ValidationStatus::Valid
        }
    }

    /// Export report to file
    pub async fn export_report(&self, report_id: &str, format: ExportFormat) -> Result<String> {
        let report = self.reports.read().unwrap()
            .get(report_id)
            .cloned()
            .ok_or_else(|| AlgoVedaError::Compliance(format!("Report not found: {}", report_id)))?;

        let filename = format!("{}_{}_{}_{}.{}",
            report.firm_identifier,
            format!("{:?}", report.report_type).to_lowercase(),
            report.period_start.format("%Y%m%d"),
            report.period_end.format("%Y%m%d"),
            format.extension()
        );

        let filepath = format!("{}/{}", self.config.output_directory, filename);

        match format {
            ExportFormat::CSV => self.export_to_csv(&report, &filepath).await?,
            ExportFormat::XML => self.export_to_xml(&report, &filepath).await?,
            ExportFormat::JSON => self.export_to_json(&report, &filepath).await?,
        }

        Ok(filepath)
    }

    /// Helper functions for export
    async fn export_to_csv(&self, report: &RegulatoryReport, filepath: &str) -> Result<()> {
        // Implementation would depend on report type
        // This is a simplified example for transaction reports
        if let ReportData::MiFID2Transaction(ref tx_report) = report.data {
            let mut wtr = WriterBuilder::new().from_path(filepath)
                .map_err(|e| AlgoVedaError::Compliance(format!("Failed to create CSV file: {}", e)))?;

            // Write headers
            wtr.write_record(&[
                "Transaction Reference",
                "Trading Date Time",
                "Instrument ID",
                "Side",
                "Quantity",
                "Price",
                "Currency",
                "Venue",
            ]).map_err(|e| AlgoVedaError::Compliance(format!("Failed to write CSV header: {}", e)))?;

            // Write data
            for tx in &tx_report.transactions {
                wtr.write_record(&[
                    &tx.transaction_reference,
                    &tx.trading_date_time.to_rfc3339(),
                    &tx.instrument_id,
                    &tx.buy_sell_indicator,
                    &tx.quantity.to_string(),
                    &tx.price.to_string(),
                    &tx.price_currency,
                    &tx.venue,
                ]).map_err(|e| AlgoVedaError::Compliance(format!("Failed to write CSV record: {}", e)))?;
            }

            wtr.flush().map_err(|e| AlgoVedaError::Compliance(format!("Failed to flush CSV file: {}", e)))?;
        }

        Ok(())
    }

    async fn export_to_xml(&self, report: &RegulatoryReport, filepath: &str) -> Result<()> {
        // XML export implementation
        let xml_content = format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<report>{}</report>", 
            serde_json::to_string(&report).unwrap());
        
        let mut file = File::create(filepath).await
            .map_err(|e| AlgoVedaError::Compliance(format!("Failed to create XML file: {}", e)))?;
        
        file.write_all(xml_content.as_bytes()).await
            .map_err(|e| AlgoVedaError::Compliance(format!("Failed to write XML file: {}", e)))?;
        
        Ok(())
    }

    async fn export_to_json(&self, report: &RegulatoryReport, filepath: &str) -> Result<()> {
        let json_content = serde_json::to_string_pretty(&report)
            .map_err(|e| AlgoVedaError::Compliance(format!("Failed to serialize report: {}", e)))?;
        
        let mut file = File::create(filepath).await
            .map_err(|e| AlgoVedaError::Compliance(format!("Failed to create JSON file: {}", e)))?;
        
        file.write_all(json_content.as_bytes()).await
            .map_err(|e| AlgoVedaError::Compliance(format!("Failed to write JSON file: {}", e)))?;
        
        Ok(())
    }

    /// Calculate concentration metrics
    fn calculate_concentration_metrics(&self, positions: &[RegulatoryPosition]) -> ConcentrationMetrics {
        let mut position_values: Vec<f64> = positions.iter().map(|p| p.market_value.abs()).collect();
        position_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let total_value: f64 = position_values.iter().sum();
        
        let largest_position_percentage = if total_value > 0.0 {
            position_values.get(0).unwrap_or(&0.0) / total_value * 100.0
        } else {
            0.0
        };

        let top_5_positions_percentage = if total_value > 0.0 {
            position_values.iter().take(5).sum::<f64>() / total_value * 100.0
        } else {
            0.0
        };

        let top_10_positions_percentage = if total_value > 0.0 {
            position_values.iter().take(10).sum::<f64>() / total_value * 100.0
        } else {
            0.0
        };

        // Calculate Herfindahl-Hirschman Index
        let hhi = if total_value > 0.0 {
            position_values.iter()
                .map(|value| {
                    let share = value / total_value;
                    share * share
                })
                .sum::<f64>() * 10000.0  // Convert to basis points
        } else {
            0.0
        };

        // Sector and geographic concentration (simplified)
        let mut sector_concentration = HashMap::new();
        let mut geographic_concentration = HashMap::new();

        for position in positions {
            let sector_value = sector_concentration.entry(position.asset_class.clone()).or_insert(0.0);
            *sector_value += position.market_value.abs();

            let geo_value = geographic_concentration.entry(position.country_code.clone()).or_insert(0.0);
            *geo_value += position.market_value.abs();
        }

        // Convert to percentages
        if total_value > 0.0 {
            for value in sector_concentration.values_mut() {
                *value = *value / total_value * 100.0;
            }
            for value in geographic_concentration.values_mut() {
                *value = *value / total_value * 100.0;
            }
        }

        ConcentrationMetrics {
            largest_position_percentage,
            top_5_positions_percentage,
            top_10_positions_percentage,
            herfindahl_hirschman_index: hhi,
            sector_concentration,
            geographic_concentration,
        }
    }

    /// Start report scheduler
    async fn start_report_scheduler(&self) {
        let report_scheduler = self.report_scheduler.clone();
        let reports = self.reports.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Check every minute
            
            loop {
                interval.tick().await;
                
                let now = Utc::now();
                let mut scheduler = report_scheduler.lock().await;
                
                // Check for due reports
                let due_reports: Vec<ScheduledReport> = scheduler.scheduled_reports
                    .range(..=now)
                    .map(|(_, report)| report.clone())
                    .collect();

                for scheduled_report in due_reports {
                    // Generate report
                    // This would call generate_report method
                    scheduler.scheduled_reports.remove(&now);
                }
            }
        });
    }

    /// Start submission processor
    async fn start_submission_processor(&self) {
        let pending_submissions = self.pending_submissions.clone();
        let reports = self.reports.clone();
        let report_events = self.report_events.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let report_id = {
                    pending_submissions.write().unwrap().pop_front()
                };
                
                if let Some(report_id) = report_id {
                    // Submit report
                    // This would implement actual submission logic
                    
                    // Update report status
                    if let Some(report) = reports.write().unwrap().get_mut(&report_id) {
                        report.submission_status = SubmissionStatus::Submitted;
                        report.submitted_at = Some(Utc::now());
                    }

                    // Emit event
                    let _ = report_events.send(ReportEvent {
                        event_id: Uuid::new_v4().to_string(),
                        report_id,
                        event_type: ReportEventType::ReportSubmitted,
                        timestamp: Utc::now(),
                        details: HashMap::new(),
                    });
                }
            }
        });
    }

    /// Start buffer cleanup
    async fn start_buffer_cleanup(&self) {
        let transaction_buffer = self.transaction_buffer.clone();
        let order_buffer = self.order_buffer.clone();
        let position_buffer = self.position_buffer.clone();
        let retention_days = self.config.archive_retention_days;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_hours(6)); // Cleanup every 6 hours
            
            loop {
                interval.tick().await;
                
                let cutoff = Utc::now() - ChronoDuration::days(retention_days as i64);
                
                // Clean transaction buffer
                {
                    let mut buffer = transaction_buffer.write().unwrap();
                    buffer.retain(|tx| tx.timestamp > cutoff);
                }
                
                // Clean order buffer
                {
                    let mut buffer = order_buffer.write().unwrap();
                    buffer.retain(|order| order.timestamp > cutoff);
                }
                
                // Clean position buffer
                {
                    let mut buffer = position_buffer.write().unwrap();
                    buffer.retain(|pos| pos.timestamp > cutoff);
                }
            }
        });
    }

    /// Get reporting statistics
    pub fn get_statistics(&self) -> ReportingStatistics {
        let reports = self.reports.read().unwrap();
        
        let mut by_type = HashMap::new();
        let mut by_jurisdiction = HashMap::new();
        let mut by_status = HashMap::new();
        
        for report in reports.values() {
            *by_type.entry(format!("{:?}", report.report_type)).or_insert(0u64) += 1;
            *by_jurisdiction.entry(format!("{:?}", report.jurisdiction)).or_insert(0u64) += 1;
            *by_status.entry(format!("{:?}", report.submission_status)).or_insert(0u64) += 1;
        }

        ReportingStatistics {
            total_reports: reports.len() as u64,
            reports_generated: self.reports_generated.load(Ordering::Relaxed),
            reports_submitted: self.reports_submitted.load(Ordering::Relaxed),
            validation_errors: self.validation_errors.load(Ordering::Relaxed),
            by_type,
            by_jurisdiction,
            by_status,
            pending_submissions: self.pending_submissions.read().unwrap().len() as u64,
        }
    }

    /// Subscribe to report events
    pub fn subscribe_events(&self) -> broadcast::Receiver<ReportEvent> {
        self.report_events.subscribe()
    }
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    CSV,
    XML,
    JSON,
}

impl ExportFormat {
    fn extension(&self) -> &str {
        match self {
            ExportFormat::CSV => "csv",
            ExportFormat::XML => "xml",
            ExportFormat::JSON => "json",
        }
    }
}

#[derive(Debug, Clone)]
struct ExecutionVenueStats {
    order_flow_percentage: f64,
    executed_volume_percentage: f64,
    average_price_improvement: f64,
    average_spread: f64,
    average_speed_of_execution: f64,
    likelihood_of_execution: f64,
    average_size_of_execution: f64,
    transaction_count: u64,
    total_volume: f64,
}

impl ExecutionVenueStats {
    fn new() -> Self {
        Self {
            order_flow_percentage: 0.0,
            executed_volume_percentage: 0.0,
            average_price_improvement: 0.0,
            average_spread: 0.0,
            average_speed_of_execution: 0.0,
            likelihood_of_execution: 0.0,
            average_size_of_execution: 0.0,
            transaction_count: 0,
            total_volume: 0.0,
        }
    }

    fn add_transaction(&mut self, transaction_data: &TransactionData) {
        self.transaction_count += 1;
        self.total_volume += transaction_data.fill.quantity as f64;
        self.average_size_of_execution = self.total_volume / self.transaction_count as f64;
        
        // Would calculate other metrics based on actual market data
        self.average_speed_of_execution = 45.0; // milliseconds
        self.average_spread = 0.01; // $0.01
        self.average_price_improvement = 0.002; // 0.2 cents
        self.likelihood_of_execution = 0.95;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingStatistics {
    pub total_reports: u64,
    pub reports_generated: u64,
    pub reports_submitted: u64,
    pub validation_errors: u64,
    pub by_type: HashMap<String, u64>,
    pub by_jurisdiction: HashMap<String, u64>,
    pub by_status: HashMap<String, u64>,
    pub pending_submissions: u64,
}

impl ReportScheduler {
    fn new() -> Self {
        Self {
            scheduled_reports: BTreeMap::new(),
            recurring_schedules: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regulatory_config_creation() {
        let config = RegulatoryConfig {
            jurisdictions: vec![Jurisdiction::US, Jurisdiction::EU],
            reporting_frequency: ReportingFrequency::Daily,
            enable_mifid2: true,
            enable_dodd_frank: true,
            enable_emir: true,
            enable_cftc: true,
            enable_finra: true,
            enable_best_execution: true,
            output_directory: "./reports".to_string(),
            archive_retention_days: 2555, // 7 years
            real_time_reporting: true,
            batch_reporting: true,
            encryption_enabled: true,
        };
        
        assert_eq!(config.jurisdictions.len(), 2);
        assert!(config.enable_mifid2);
    }

    #[tokio::test]
    async fn test_concentration_metrics_calculation() {
        let config = RegulatoryConfig {
            jurisdictions: vec![Jurisdiction::US],
            reporting_frequency: ReportingFrequency::Daily,
            enable_mifid2: false,
            enable_dodd_frank: true,
            enable_emir: false,
            enable_cftc: false,
            enable_finra: true,
            enable_best_execution: true,
            output_directory: "./reports".to_string(),
            archive_retention_days: 30,
            real_time_reporting: false,
            batch_reporting: true,
            encryption_enabled: false,
        };

        let engine = RegulatoryReportingEngine::new(config);
        
        let positions = vec![
            RegulatoryPosition {
                instrument_id: "AAPL".to_string(),
                instrument_type: "EQUITY".to_string(),
                position_date: Utc::now().date_naive(),
                long_quantity: 1000.0,
                short_quantity: 0.0,
                net_quantity: 1000.0,
                market_value: 150000.0,
                currency: "USD".to_string(),
                venue: "NASDAQ".to_string(),
                asset_class: "EQUITY".to_string(),
                country_code: "US".to_string(),
                issuer_lei: None,
            },
            RegulatoryPosition {
                instrument_id: "MSFT".to_string(),
                instrument_type: "EQUITY".to_string(),
                position_date: Utc::now().date_naive(),
                long_quantity: 500.0,
                short_quantity: 0.0,
                net_quantity: 500.0,
                market_value: 100000.0,
                currency: "USD".to_string(),
                venue: "NASDAQ".to_string(),
                asset_class: "EQUITY".to_string(),
                country_code: "US".to_string(),
                issuer_lei: None,
            },
        ];

        let metrics = engine.calculate_concentration_metrics(&positions);
        
        assert_eq!(metrics.largest_position_percentage, 60.0); // 150k out of 250k total
        assert!(metrics.herfindahl_hirschman_index > 0.0);
    }
}
