"""
FIX Protocol Gateway
Professional FIX 4.2/4.4/5.0 gateway for institutional connectivity
"""

import asyncio
import socket
import ssl
import struct
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
import logging
import json

import quickfix as fix
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class FIXVersion(Enum):
    FIX42 = "FIX.4.2"
    FIX44 = "FIX.4.4"
    FIX50 = "FIX.5.0"
    FIX50SP2 = "FIX.5.0SP2"

class FIXMessageType(Enum):
    HEARTBEAT = "0"
    TEST_REQUEST = "1"
    RESEND_REQUEST = "2"
    REJECT = "3"
    SEQUENCE_RESET = "4"
    LOGOUT = "5"
    IOI = "6"
    ADVERTISEMENT = "7"
    EXECUTION_REPORT = "8"
    ORDER_CANCEL_REJECT = "9"
    LOGON = "A"
    NEWS = "B"
    EMAIL = "C"
    NEW_ORDER_SINGLE = "D"
    ORDER_CANCEL_REQUEST = "F"
    ORDER_CANCEL_REPLACE_REQUEST = "G"
    ORDER_STATUS_REQUEST = "H"
    ALLOCATION_INSTRUCTION = "J"
    LIST_CANCEL_REQUEST = "K"
    LIST_EXECUTE = "L"
    LIST_STATUS_REQUEST = "M"
    NEW_ORDER_LIST = "E"
    QUOTE_REQUEST = "R"
    QUOTE = "S"
    SETTLEMENT_INSTRUCTIONS = "T"
    MARKET_DATA_REQUEST = "V"
    MARKET_DATA_SNAPSHOT_FULL_REFRESH = "W"
    MARKET_DATA_INCREMENTAL_REFRESH = "X"
    MARKET_DATA_REQUEST_REJECT = "Y"
    QUOTE_CANCEL = "Z"
    QUOTE_STATUS_REQUEST = "a"
    MASS_QUOTE_ACKNOWLEDGEMENT = "b"
    SECURITY_DEFINITION_REQUEST = "c"
    SECURITY_DEFINITION = "d"
    SECURITY_STATUS_REQUEST = "e"
    SECURITY_STATUS = "f"
    TRADING_SESSION_STATUS_REQUEST = "g"
    TRADING_SESSION_STATUS = "h"
    MASS_QUOTE = "i"
    BUSINESS_MESSAGE_REJECT = "j"

@dataclass
class FIXConfig:
    version: FIXVersion = FIXVersion.FIX44
    sender_comp_id: str = "ALGOVEDA"
    target_comp_id: str = "EXCHANGE"
    host: str = "localhost"
    port: int = 9876
    heartbeat_interval: int = 30
    logon_timeout: int = 10
    logout_timeout: int = 5
    username: Optional[str] = None
    password: Optional[str] = None
    reset_on_logon: bool = False
    reset_on_logout: bool = False
    reset_on_disconnect: bool = False
    validate_length_and_checksum: bool = True
    validate_fields_out_of_order: bool = True
    validate_fields_have_values: bool = True
    validate_user_defined_fields: bool = True
    allow_unknown_msg_fields: bool = False
    preserve_message_fields_order: bool = False
    check_latent_admin_messages: bool = True
    max_latency_seconds: int = 120
    send_redundant_resend_requests: bool = False
    resend_request_chunk_size: int = 0
    enable_last_msg_seq_num_processed: bool = False
    socket_accept_host: str = ""
    socket_connect_host: str = ""
    socket_connect_port: int = 0
    socket_nodelay: bool = True
    socket_send_buffer_size: int = 0
    socket_receive_buffer_size: int = 0
    persist_messages: bool = True
    file_store_path: str = "./fix_store"
    file_log_path: str = "./fix_logs"
    use_ssl: bool = False
    ssl_certificate: Optional[str] = None
    ssl_private_key: Optional[str] = None
    ssl_ca_certificate: Optional[str] = None

@dataclass
class FIXMessage:
    msg_type: FIXMessageType
    fields: Dict[int, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_message: str = ""
    sequence_number: int = 0
    sender_comp_id: str = ""
    target_comp_id: str = ""
    
class FIXSession:
    def __init__(self, session_id: fix.SessionID, config: FIXConfig):
        self.session_id = session_id
        self.config = config
        self.is_logged_on = False
        self.last_heartbeat = time.time()
        self.expected_target_num = 1
        self.next_sender_msg_seq_num = 1
        self.message_store: Dict[int, FIXMessage] = {}
        self.pending_resend_requests: Set[int] = set()
        self.application_callbacks: Dict[FIXMessageType, List[Callable]] = defaultdict(list)
        
        # Performance metrics
        self.messages_sent = Counter('fix_messages_sent_total', 'Total FIX messages sent', ['session', 'msg_type'])
        self.messages_received = Counter('fix_messages_received_total', 'Total FIX messages received', ['session', 'msg_type'])
        self.message_latency = Histogram('fix_message_latency_seconds', 'FIX message processing latency', ['session', 'msg_type'])
        self.active_sessions = Gauge('fix_active_sessions', 'Number of active FIX sessions')
        
    def is_session_time(self) -> bool:
        """Check if current time is within session hours"""
        # Simplified - would implement proper session time logic
        return True
    
    def should_send_logon(self) -> bool:
        """Determine if logon should be sent"""
        return not self.is_logged_on and self.is_session_time()
    
    def should_send_logout(self) -> bool:
        """Determine if logout should be sent"""
        return self.is_logged_on and not self.is_session_time()
    
    def register_callback(self, msg_type: FIXMessageType, callback: Callable[[FIXMessage], None]):
        """Register callback for specific message type"""
        self.application_callbacks[msg_type].append(callback)
    
    def process_message(self, message: FIXMessage):
        """Process incoming FIX message"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.messages_received.labels(
                session=str(self.session_id),
                msg_type=message.msg_type.value
            ).inc()
            
            # Handle administrative messages
            if message.msg_type == FIXMessageType.HEARTBEAT:
                self.last_heartbeat = time.time()
            elif message.msg_type == FIXMessageType.LOGON:
                self.handle_logon(message)
            elif message.msg_type == FIXMessageType.LOGOUT:
                self.handle_logout(message)
            elif message.msg_type == FIXMessageType.RESEND_REQUEST:
                self.handle_resend_request(message)
            elif message.msg_type == FIXMessageType.SEQUENCE_RESET:
                self.handle_sequence_reset(message)
            else:
                # Call application callbacks
                for callback in self.application_callbacks[message.msg_type]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in application callback: {e}")
            
            # Store message
            if self.config.persist_messages:
                self.message_store[message.sequence_number] = message
            
        finally:
            # Record processing latency
            latency = time.time() - start_time
            self.message_latency.labels(
                session=str(self.session_id),
                msg_type=message.msg_type.value
            ).observe(latency)
    
    def handle_logon(self, message: FIXMessage):
        """Handle logon message"""
        self.is_logged_on = True
        self.active_sessions.inc()
        logger.info(f"Session {self.session_id} logged on")
    
    def handle_logout(self, message: FIXMessage):
        """Handle logout message"""
        self.is_logged_on = False
        self.active_sessions.dec()
        logger.info(f"Session {self.session_id} logged out")
    
    def handle_resend_request(self, message: FIXMessage):
        """Handle resend request"""
        begin_seq_no = int(message.fields.get(7, "0"))  # BeginSeqNo
        end_seq_no = int(message.fields.get(16, "0"))   # EndSeqNo
        
        # Resend requested messages
        for seq_num in range(begin_seq_no, end_seq_no + 1):
            if seq_num in self.message_store:
                # Resend message
                stored_msg = self.message_store[seq_num]
                self.send_message(stored_msg)
    
    def handle_sequence_reset(self, message: FIXMessage):
        """Handle sequence reset"""
        new_seq_no = int(message.fields.get(36, "0"))  # NewSeqNo
        gap_fill_flag = message.fields.get(123, "N")    # GapFillFlag
        
        if gap_fill_flag == "Y":
            # Gap fill sequence reset
            self.expected_target_num = new_seq_no
        else:
            # Hard sequence reset
            self.expected_target_num = new_seq_no
            self.message_store.clear()
    
    def send_message(self, message: FIXMessage) -> bool:
        """Send FIX message"""
        try:
            # Update sequence number
            message.sequence_number = self.next_sender_msg_seq_num
            self.next_sender_msg_seq_num += 1
            
            # Update metrics
            self.messages_sent.labels(
                session=str(self.session_id),
                msg_type=message.msg_type.value
            ).inc()
            
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

class FIXEngine:
    """High-performance FIX protocol engine"""
    
    def __init__(self, config: FIXConfig):
        self.config = config
        self.sessions: Dict[str, FIXSession] = {}
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.running = False
        self.stats = {
            'messages_processed': 0,
            'sessions_active': 0,
            'uptime_start': time.time(),
            'last_heartbeat': time.time()
        }
        
        # Initialize QuickFIX components
        self.settings = self._create_settings()
        self.application = FIXApplication(self)
        self.message_store_factory = fix.FileStoreFactory(self.settings)
        self.log_factory = fix.FileLogFactory(self.settings)
    
    def _create_settings(self) -> fix.Settings:
        """Create QuickFIX settings"""
        settings = fix.Settings()
        
        # Default session settings
        dictionary = fix.Dictionary()
        dictionary.setString(fix.CONNECTION_TYPE, "initiator")
        dictionary.setString(fix.BEGIN_STRING, self.config.version.value)
        dictionary.setString(fix.SENDER_COMP_ID, self.config.sender_comp_id)
        dictionary.setString(fix.TARGET_COMP_ID, self.config.target_comp_id)
        dictionary.setString(fix.SOCKET_CONNECT_HOST, self.config.host)
        dictionary.setInt(fix.SOCKET_CONNECT_PORT, self.config.port)
        dictionary.setInt(fix.HEARTBTINT, self.config.heartbeat_interval)
        dictionary.setString(fix.START_TIME, "00:00:00")
        dictionary.setString(fix.END_TIME, "00:00:00")
        dictionary.setBool(fix.USE_DATA_DICTIONARY, True)
        dictionary.setBool(fix.VALIDATE_LENGTH_AND_CHECKSUM, self.config.validate_length_and_checksum)
        dictionary.setBool(fix.VALIDATE_FIELDS_OUT_OF_ORDER, self.config.validate_fields_out_of_order)
        dictionary.setBool(fix.VALIDATE_FIELDS_HAVE_VALUES, self.config.validate_fields_have_values)
        dictionary.setBool(fix.VALIDATE_USER_DEFINED_FIELDS, self.config.validate_user_defined_fields)
        dictionary.setBool(fix.ALLOW_UNKNOWN_MSG_FIELDS, self.config.allow_unknown_msg_fields)
        dictionary.setBool(fix.PRESERVE_MESSAGE_FIELDS_ORDER, self.config.preserve_message_fields_order)
        dictionary.setBool(fix.CHECK_LATENT_ADMIN_MESSAGES, self.config.check_latent_admin_messages)
        dictionary.setInt(fix.MAX_LATENCY, self.config.max_latency_seconds)
        dictionary.setBool(fix.SEND_REDUNDANT_RESEND_REQUESTS, self.config.send_redundant_resend_requests)
        dictionary.setInt(fix.RESEND_REQUEST_CHUNK_SIZE, self.config.resend_request_chunk_size)
        dictionary.setBool(fix.ENABLE_LAST_MSG_SEQ_NUM_PROCESSED, self.config.enable_last_msg_seq_num_processed)
        dictionary.setBool(fix.SOCKET_NODELAY, self.config.socket_nodelay)
        dictionary.setString(fix.FILE_STORE_PATH, self.config.file_store_path)
        dictionary.setString(fix.FILE_LOG_PATH, self.config.file_log_path)
        
        if self.config.socket_send_buffer_size > 0:
            dictionary.setInt(fix.SOCKET_SEND_BUFFER_SIZE, self.config.socket_send_buffer_size)
        
        if self.config.socket_receive_buffer_size > 0:
            dictionary.setInt(fix.SOCKET_RECEIVE_BUFFER_SIZE, self.config.socket_receive_buffer_size)
        
        if self.config.username:
            dictionary.setString(fix.USERNAME, self.config.username)
        
        if self.config.password:
            dictionary.setString(fix.PASSWORD, self.config.password)
        
        # SSL configuration
        if self.config.use_ssl:
            dictionary.setBool(fix.SOCKET_USE_SSL, True)
            if self.config.ssl_certificate:
                dictionary.setString(fix.SOCKET_CERTIFICATE_FILE, self.config.ssl_certificate)
            if self.config.ssl_private_key:
                dictionary.setString(fix.SOCKET_PRIVATE_KEY_FILE, self.config.ssl_private_key)
            if self.config.ssl_ca_certificate:
                dictionary.setString(fix.SOCKET_CA_CERTIFICATE_FILE, self.config.ssl_ca_certificate)
        
        session_id = fix.SessionID(
            self.config.version.value,
            self.config.sender_comp_id,
            self.config.target_comp_id
        )
        
        settings.set(session_id, dictionary)
        return settings
    
    async def start(self):
        """Start the FIX engine"""
        self.running = True
        
        try:
            # Create initiator
            self.initiator = fix.SocketInitiator(
                self.application,
                self.message_store_factory,
                self.settings,
                self.log_factory
            )
            
            # Start QuickFIX engine
            self.initiator.start()
            
            # Start message processing loop
            asyncio.create_task(self._process_messages())
            
            # Start heartbeat task
            asyncio.create_task(self._heartbeat_task())
            
            logger.info("FIX engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start FIX engine: {e}")
            raise
    
    async def stop(self):
        """Stop the FIX engine"""
        self.running = False
        
        if hasattr(self, 'initiator'):
            self.initiator.stop()
        
        logger.info("FIX engine stopped")
    
    async def send_new_order_single(self, order_data: Dict[str, Any]) -> str:
        """Send New Order Single (D) message"""
        message = fix.Message()
        header = message.getHeader()
        
        # Set message type
        header.setField(fix.MsgType(FIXMessageType.NEW_ORDER_SINGLE.value))
        
        # Required fields
        message.setField(fix.ClOrdID(order_data['client_order_id']))
        message.setField(fix.Symbol(order_data['symbol']))
        message.setField(fix.Side(order_data['side']))  # 1=Buy, 2=Sell
        message.setField(fix.OrdType(order_data['order_type']))  # 1=Market, 2=Limit
        message.setField(fix.OrderQty(order_data['quantity']))
        message.setField(fix.TransactTime(fix.UtcTimeStamp()))
        
        # Optional fields
        if 'price' in order_data:
            message.setField(fix.Price(order_data['price']))
        
        if 'time_in_force' in order_data:
            message.setField(fix.TimeInForce(order_data['time_in_force']))
        
        if 'account' in order_data:
            message.setField(fix.Account(order_data['account']))
        
        # Send message
        session_id = fix.SessionID(
            self.config.version.value,
            self.config.sender_comp_id,
            self.config.target_comp_id
        )
        
        success = fix.Session.sendToTarget(message, session_id)
        if success:
            logger.info(f"Sent New Order Single: {order_data['client_order_id']}")
            return order_data['client_order_id']
        else:
            raise Exception("Failed to send New Order Single")
    
    async def send_order_cancel_request(self, cancel_data: Dict[str, Any]) -> str:
        """Send Order Cancel Request (F) message"""
        message = fix.Message()
        header = message.getHeader()
        
        # Set message type
        header.setField(fix.MsgType(FIXMessageType.ORDER_CANCEL_REQUEST.value))
        
        # Required fields
        message.setField(fix.OrigClOrdID(cancel_data['original_client_order_id']))
        message.setField(fix.ClOrdID(cancel_data['client_order_id']))
        message.setField(fix.Symbol(cancel_data['symbol']))
        message.setField(fix.Side(cancel_data['side']))
        message.setField(fix.TransactTime(fix.UtcTimeStamp()))
        
        # Send message
        session_id = fix.SessionID(
            self.config.version.value,
            self.config.sender_comp_id,
            self.config.target_comp_id
        )
        
        success = fix.Session.sendToTarget(message, session_id)
        if success:
            logger.info(f"Sent Order Cancel Request: {cancel_data['client_order_id']}")
            return cancel_data['client_order_id']
        else:
            raise Exception("Failed to send Order Cancel Request")
    
    async def send_market_data_request(self, request_data: Dict[str, Any]) -> str:
        """Send Market Data Request (V) message"""
        message = fix.Message()
        header = message.getHeader()
        
        # Set message type
        header.setField(fix.MsgType(FIXMessageType.MARKET_DATA_REQUEST.value))
        
        # Required fields
        md_req_id = request_data.get('md_req_id', f"MDR_{int(time.time())}")
        message.setField(fix.MDReqID(md_req_id))
        message.setField(fix.SubscriptionRequestType(request_data['subscription_type']))  # 0=Snapshot, 1=Subscribe, 2=Unsubscribe
        message.setField(fix.MarketDepth(request_data.get('market_depth', 1)))
        
        # Entry types group
        entry_types = request_data.get('entry_types', ['0', '1'])  # 0=Bid, 1=Ask
        group = fix.Group(fix.NoMDEntryTypes(), fix.MDEntryType())
        for entry_type in entry_types:
            group.setField(fix.MDEntryType(entry_type))
            message.addGroup(group)
        
        # Symbols group
        symbols = request_data.get('symbols', [])
        symbol_group = fix.Group(fix.NoRelatedSym(), fix.Symbol())
        for symbol in symbols:
            symbol_group.setField(fix.Symbol(symbol))
            message.addGroup(symbol_group)
        
        # Send message
        session_id = fix.SessionID(
            self.config.version.value,
            self.config.sender_comp_id,
            self.config.target_comp_id
        )
        
        success = fix.Session.sendToTarget(message, session_id)
        if success:
            logger.info(f"Sent Market Data Request: {md_req_id}")
            return md_req_id
        else:
            raise Exception("Failed to send Market Data Request")
    
    async def _process_messages(self):
        """Process incoming messages from queue"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process the message
                await self._handle_message(message)
                self.stats['messages_processed'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, fix_message):
        """Handle incoming FIX message"""
        try:
            # Parse message
            msg_type = fix_message.getHeader().getField(fix.MsgType())
            
            # Convert to internal format
            message = FIXMessage(
                msg_type=FIXMessageType(msg_type),
                raw_message=fix_message.toString(),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Extract common fields
            if fix_message.isSetField(fix.MsgSeqNum()):
                message.sequence_number = fix_message.getField(fix.MsgSeqNum())
            
            if fix_message.getHeader().isSetField(fix.SenderCompID()):
                message.sender_comp_id = fix_message.getHeader().getField(fix.SenderCompID())
            
            if fix_message.getHeader().isSetField(fix.TargetCompID()):
                message.target_comp_id = fix_message.getHeader().getField(fix.TargetCompID())
            
            # Handle specific message types
            if message.msg_type == FIXMessageType.EXECUTION_REPORT:
                await self._handle_execution_report(fix_message, message)
            elif message.msg_type == FIXMessageType.ORDER_CANCEL_REJECT:
                await self._handle_order_cancel_reject(fix_message, message)
            elif message.msg_type == FIXMessageType.MARKET_DATA_SNAPSHOT_FULL_REFRESH:
                await self._handle_market_data_snapshot(fix_message, message)
            elif message.msg_type == FIXMessageType.MARKET_DATA_INCREMENTAL_REFRESH:
                await self._handle_market_data_incremental(fix_message, message)
            
        except Exception as e:
            logger.error(f"Error handling FIX message: {e}")
    
    async def _handle_execution_report(self, fix_message, message: FIXMessage):
        """Handle Execution Report (8) message"""
        try:
            # Extract execution report fields
            exec_report = {
                'order_id': fix_message.getField(fix.OrderID()) if fix_message.isSetField(fix.OrderID()) else None,
                'client_order_id': fix_message.getField(fix.ClOrdID()),
                'exec_id': fix_message.getField(fix.ExecID()),
                'exec_type': fix_message.getField(fix.ExecType()),
                'ord_status': fix_message.getField(fix.OrdStatus()),
                'symbol': fix_message.getField(fix.Symbol()),
                'side': fix_message.getField(fix.Side()),
                'leaves_qty': float(fix_message.getField(fix.LeavesQty())) if fix_message.isSetField(fix.LeavesQty()) else 0.0,
                'cum_qty': float(fix_message.getField(fix.CumQty())) if fix_message.isSetField(fix.CumQty()) else 0.0,
                'avg_px': float(fix_message.getField(fix.AvgPx())) if fix_message.isSetField(fix.AvgPx()) else 0.0,
            }
            
            if fix_message.isSetField(fix.LastQty()):
                exec_report['last_qty'] = float(fix_message.getField(fix.LastQty()))
            
            if fix_message.isSetField(fix.LastPx()):
                exec_report['last_px'] = float(fix_message.getField(fix.LastPx()))
            
            if fix_message.isSetField(fix.TransactTime()):
                exec_report['transact_time'] = fix_message.getField(fix.TransactTime())
            
            # Notify application
            await self._notify_execution_report(exec_report)
            
        except Exception as e:
            logger.error(f"Error processing execution report: {e}")
    
    async def _handle_order_cancel_reject(self, fix_message, message: FIXMessage):
        """Handle Order Cancel Reject (9) message"""
        try:
            cancel_reject = {
                'order_id': fix_message.getField(fix.OrderID()) if fix_message.isSetField(fix.OrderID()) else None,
                'client_order_id': fix_message.getField(fix.ClOrdID()),
                'orig_client_order_id': fix_message.getField(fix.OrigClOrdID()),
                'ord_status': fix_message.getField(fix.OrdStatus()),
                'cxl_rej_response_to': fix_message.getField(fix.CxlRejResponseTo()),
                'cxl_rej_reason': fix_message.getField(fix.CxlRejReason()) if fix_message.isSetField(fix.CxlRejReason()) else None,
                'text': fix_message.getField(fix.Text()) if fix_message.isSetField(fix.Text()) else None,
            }
            
            # Notify application
            await self._notify_cancel_reject(cancel_reject)
            
        except Exception as e:
            logger.error(f"Error processing cancel reject: {e}")
    
    async def _handle_market_data_snapshot(self, fix_message, message: FIXMessage):
        """Handle Market Data Snapshot Full Refresh (W) message"""
        try:
            snapshot = {
                'md_req_id': fix_message.getField(fix.MDReqID()) if fix_message.isSetField(fix.MDReqID()) else None,
                'symbol': fix_message.getField(fix.Symbol()),
                'entries': []
            }
            
            # Extract market data entries
            no_md_entries = fix.NoMDEntries()
            if fix_message.isSetField(no_md_entries):
                num_entries = fix_message.getField(no_md_entries)
                
                for i in range(1, num_entries + 1):
                    group = fix.Group(no_md_entries.getField(), fix.MDEntryType().getField())
                    fix_message.getGroup(i, group)
                    
                    entry = {
                        'md_entry_type': group.getField(fix.MDEntryType()),
                        'md_entry_px': float(group.getField(fix.MDEntryPx())) if group.isSetField(fix.MDEntryPx()) else None,
                        'md_entry_size': float(group.getField(fix.MDEntrySize())) if group.isSetField(fix.MDEntrySize()) else None,
                        'md_entry_time': group.getField(fix.MDEntryTime()) if group.isSetField(fix.MDEntryTime()) else None,
                    }
                    
                    snapshot['entries'].append(entry)
            
            # Notify application
            await self._notify_market_data_snapshot(snapshot)
            
        except Exception as e:
            logger.error(f"Error processing market data snapshot: {e}")
    
    async def _handle_market_data_incremental(self, fix_message, message: FIXMessage):
        """Handle Market Data Incremental Refresh (X) message"""
        try:
            incremental = {
                'md_req_id': fix_message.getField(fix.MDReqID()) if fix_message.isSetField(fix.MDReqID()) else None,
                'entries': []
            }
            
            # Extract incremental entries
            no_md_entries = fix.NoMDEntries()
            if fix_message.isSetField(no_md_entries):
                num_entries = fix_message.getField(no_md_entries)
                
                for i in range(1, num_entries + 1):
                    group = fix.Group(no_md_entries.getField(), fix.MDEntryType().getField())
                    fix_message.getGroup(i, group)
                    
                    entry = {
                        'md_update_action': group.getField(fix.MDUpdateAction()) if group.isSetField(fix.MDUpdateAction()) else None,
                        'md_entry_type': group.getField(fix.MDEntryType()),
                        'symbol': group.getField(fix.Symbol()) if group.isSetField(fix.Symbol()) else None,
                        'md_entry_px': float(group.getField(fix.MDEntryPx())) if group.isSetField(fix.MDEntryPx()) else None,
                        'md_entry_size': float(group.getField(fix.MDEntrySize())) if group.isSetField(fix.MDEntrySize()) else None,
                    }
                    
                    incremental['entries'].append(entry)
            
            # Notify application
            await self._notify_market_data_incremental(incremental)
            
        except Exception as e:
            logger.error(f"Error processing incremental market data: {e}")
    
    async def _heartbeat_task(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Send test request if no recent activity
                current_time = time.time()
                if current_time - self.stats['last_heartbeat'] > self.config.heartbeat_interval * 2:
                    await self._send_test_request()
                
                self.stats['last_heartbeat'] = current_time
                
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
    
    async def _send_test_request(self):
        """Send Test Request (1) message"""
        message = fix.Message()
        header = message.getHeader()
        
        header.setField(fix.MsgType(FIXMessageType.TEST_REQUEST.value))
        message.setField(fix.TestReqID(f"TEST_{int(time.time())}"))
        
        session_id = fix.SessionID(
            self.config.version.value,
            self.config.sender_comp_id,
            self.config.target_comp_id
        )
        
        fix.Session.sendToTarget(message, session_id)
        logger.debug("Sent test request")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        uptime = time.time() - self.stats['uptime_start']
        
        return {
            'uptime_seconds': uptime,
            'messages_processed': self.stats['messages_processed'],
            'sessions_active': len([s for s in self.sessions.values() if s.is_logged_on]),
            'message_rate': self.stats['messages_processed'] / max(uptime, 1),
            'last_heartbeat': self.stats['last_heartbeat'],
            'config': {
                'version': self.config.version.value,
                'sender_comp_id': self.config.sender_comp_id,
                'target_comp_id': self.config.target_comp_id,
                'host': self.config.host,
                'port': self.config.port,
            }
        }
    
    # Application callback methods (to be overridden by users)
    async def _notify_execution_report(self, exec_report: Dict[str, Any]):
        """Override this method to handle execution reports"""
        logger.info(f"Execution Report: {exec_report}")
    
    async def _notify_cancel_reject(self, cancel_reject: Dict[str, Any]):
        """Override this method to handle cancel rejects"""
        logger.info(f"Cancel Reject: {cancel_reject}")
    
    async def _notify_market_data_snapshot(self, snapshot: Dict[str, Any]):
        """Override this method to handle market data snapshots"""
        logger.debug(f"Market Data Snapshot: {snapshot['symbol']}")
    
    async def _notify_market_data_incremental(self, incremental: Dict[str, Any]):
        """Override this method to handle incremental market data"""
        logger.debug(f"Market Data Incremental: {len(incremental['entries'])} entries")

class FIXApplication(fix.Application):
    """QuickFIX Application implementation"""
    
    def __init__(self, engine: FIXEngine):
        super().__init__()
        self.engine = engine
    
    def onCreate(self, sessionID):
        """Called when session is created"""
        logger.info(f"Session created: {sessionID}")
        session = FIXSession(sessionID, self.engine.config)
        self.engine.sessions[str(sessionID)] = session
    
    def onLogon(self, sessionID):
        """Called on successful logon"""
        logger.info(f"Logon: {sessionID}")
        if str(sessionID) in self.engine.sessions:
            self.engine.sessions[str(sessionID)].is_logged_on = True
    
    def onLogout(self, sessionID):
        """Called on logout"""
        logger.info(f"Logout: {sessionID}")
        if str(sessionID) in self.engine.sessions:
            self.engine.sessions[str(sessionID)].is_logged_on = False
    
    def toAdmin(self, message, sessionID):
        """Called for outgoing admin messages"""
        msg_type = message.getHeader().getField(fix.MsgType())
        
        # Add authentication for logon messages
        if msg_type == FIXMessageType.LOGON.value:
            if self.engine.config.username:
                message.setField(fix.Username(self.engine.config.username))
            if self.engine.config.password:
                message.setField(fix.Password(self.engine.config.password))
    
    def fromAdmin(self, message, sessionID):
        """Called for incoming admin messages"""
        # Queue admin message for processing
        asyncio.create_task(self.engine.message_queue.put(message))
    
    def toApp(self, message, sessionID):
        """Called for outgoing application messages"""
        logger.debug(f"Sending app message: {message.getHeader().getField(fix.MsgType())}")
    
    def fromApp(self, message, sessionID):
        """Called for incoming application messages"""
        # Queue application message for processing
        asyncio.create_task(self.engine.message_queue.put(message))

# Example usage and testing
async def example_usage():
    """Example of how to use the FIX gateway"""
    
    # Configure FIX connection
    config = FIXConfig(
        version=FIXVersion.FIX44,
        sender_comp_id="ALGOVEDA",
        target_comp_id="EXCHANGE",
        host="fix.exchange.com",
        port=9876,
        username="trader1",
        password="password123",
        heartbeat_interval=30
    )
    
    # Create and start FIX engine
    engine = FIXEngine(config)
    
    # Override callback methods
    class CustomFIXEngine(FIXEngine):
        async def _notify_execution_report(self, exec_report):
            print(f"Order {exec_report['client_order_id']} executed: {exec_report['last_qty']} @ {exec_report['last_px']}")
        
        async def _notify_market_data_snapshot(self, snapshot):
            print(f"Market data for {snapshot['symbol']}: {len(snapshot['entries'])} levels")
    
    custom_engine = CustomFIXEngine(config)
    
    try:
        await custom_engine.start()
        
        # Send a new order
        await custom_engine.send_new_order_single({
            'client_order_id': 'ORDER_001',
            'symbol': 'AAPL',
            'side': '1',  # Buy
            'order_type': '2',  # Limit
            'quantity': 100,
            'price': 150.00,
            'time_in_force': '0',  # Day
        })
        
        # Subscribe to market data
        await custom_engine.send_market_data_request({
            'subscription_type': '1',  # Subscribe
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'entry_types': ['0', '1'],  # Bid, Ask
            'market_depth': 5
        })
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            stats = custom_engine.get_statistics()
            if stats['messages_processed'] % 100 == 0:
                print(f"Processed {stats['messages_processed']} messages, uptime: {stats['uptime_seconds']:.1f}s")
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await custom_engine.stop()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
