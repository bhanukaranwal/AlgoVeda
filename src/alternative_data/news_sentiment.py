"""
News Sentiment Analysis Engine
Real-time news ingestion, sentiment analysis, and alpha signal generation
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import hashlib
from collections import defaultdict, deque

import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from newspaper import Article
import feedparser
import websocket
import threading
import time

logger = logging.getLogger(__name__)

class NewsSource(Enum):
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    CNBC = "cnbc"
    WSJ = "wsj"
    MARKETWATCH = "marketwatch"
    SEEKING_ALPHA = "seeking_alpha"
    YAHOO_FINANCE = "yahoo_finance"
    GOOGLE_NEWS = "google_news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    CUSTOM_RSS = "custom_rss"
    WEBHOOK = "webhook"

class SentimentModel(Enum):
    TEXTBLOB = "textblob"
    VADER = "vader"
    FINBERT = "finbert"
    ROBERTA = "roberta"
    CUSTOM_TRANSFORMER = "custom_transformer"

class NewsCategory(Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    ANALYST_RATING = "analyst_rating"
    REGULATORY = "regulatory"
    MARKET_MOVING = "market_moving"
    TECHNICAL = "technical"
    ECONOMIC_DATA = "economic_data"
    COMPANY_SPECIFIC = "company_specific"
    SECTOR_NEWS = "sector_news"
    GENERAL_MARKET = "general_market"

@dataclass
class NewsConfig:
    sources: List[NewsSource] = field(default_factory=lambda: [NewsSource.REUTERS, NewsSource.BLOOMBERG])
    sentiment_models: List[SentimentModel] = field(default_factory=lambda: [SentimentModel.FINBERT])
    symbols_to_track: Set[str] = field(default_factory=set)
    sectors_to_track: Set[str] = field(default_factory=set)
    keywords_to_track: Set[str] = field(default_factory=set)
    languages: Set[str] = field(default_factory=lambda: {"en"})
    max_articles_per_hour: int = 1000
    sentiment_threshold: float = 0.1
    relevance_threshold: float = 0.3
    enable_real_time_processing: bool = True
    enable_historical_analysis: bool = True
    cache_duration_hours: int = 24
    max_cache_size: int = 100000
    api_keys: Dict[str, str] = field(default_factory=dict)
    webhook_endpoints: List[str] = field(default_factory=list)
    output_format: str = "json"  # json, csv, database
    enable_entity_extraction: bool = True
    enable_topic_modeling: bool = True
    enable_anomaly_detection: bool = True

@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    summary: str
    source: NewsSource
    url: str
    published_at: datetime
    processed_at: datetime
    author: Optional[str] = None
    symbols_mentioned: Set[str] = field(default_factory=set)
    entities: List[str] = field(default_factory=list)
    categories: List[NewsCategory] = field(default_factory=list)
    sentiment_scores: Dict[str, float] = field(default_factory=dict)
    relevance_score: float = 0.0
    impact_score: float = 0.0
    language: str = "en"
    word_count: int = 0
    reading_time_minutes: float = 0.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SentimentAnalysis:
    model_name: str
    overall_sentiment: str  # positive, negative, neutral
    confidence: float
    sentiment_score: float  # -1 to 1
    positive_score: float
    negative_score: float
    neutral_score: float
    compound_score: float
    emotions: Dict[str, float] = field(default_factory=dict)
    aspects: Dict[str, float] = field(default_factory=dict)  # aspect-based sentiment
    
@dataclass
class NewsSignal:
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # momentum, reversal, event_driven
    signal_strength: float  # -1 to 1
    confidence: float  # 0 to 1
    news_articles: List[str]  # article IDs
    sentiment_aggregate: float
    volume_impact_expected: float
    price_impact_expected: float
    time_horizon_hours: int
    risk_level: str  # low, medium, high
    metadata: Dict[str, Any] = field(default_factory=dict)

class NewsIngestionEngine:
    """Handles ingestion from multiple news sources"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        self.session = None
        self.rss_feeds = self._initialize_rss_feeds()
        self.api_endpoints = self._initialize_api_endpoints()
        self.rate_limiters = defaultdict(lambda: deque(maxlen=100))
        self.processed_articles = set()  # Deduplication
        
    def _initialize_rss_feeds(self) -> Dict[NewsSource, List[str]]:
        """Initialize RSS feed URLs for different sources"""
        return {
            NewsSource.REUTERS: [
                "http://feeds.reuters.com/reuters/businessNews",
                "http://feeds.reuters.com/reuters/companyNews",
                "http://feeds.reuters.com/reuters/marketsNews"
            ],
            NewsSource.CNBC: [
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                "https://www.cnbc.com/id/10001147/device/rss/rss.html"
            ],
            NewsSource.MARKETWATCH: [
                "http://feeds.marketwatch.com/marketwatch/marketpulse",
                "http://feeds.marketwatch.com/marketwatch/realtimeheadlines"
            ],
            NewsSource.YAHOO_FINANCE: [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://feeds.finance.yahoo.com/rss/2.0/category-stocks"
            ]
        }
    
    def _initialize_api_endpoints(self) -> Dict[NewsSource, Dict[str, str]]:
        """Initialize API endpoints for premium sources"""
        return {
            NewsSource.BLOOMBERG: {
                "base_url": "https://api.bloomberg.com/v1/news",
                "headers": {"X-API-Key": self.config.api_keys.get("bloomberg", "")}
            },
            NewsSource.REUTERS: {
                "base_url": "https://api.reuters.com/v1/news",
                "headers": {"Authorization": f"Bearer {self.config.api_keys.get('reuters', '')}"}
            }
        }
    
    async def start(self):
        """Start the news ingestion engine"""
        self.session = aiohttp.ClientSession()
        
        # Start RSS monitoring
        asyncio.create_task(self._monitor_rss_feeds())
        
        # Start API polling
        asyncio.create_task(self._poll_api_endpoints())
        
        # Start webhook server
        if self.config.webhook_endpoints:
            asyncio.create_task(self._start_webhook_server())
    
    async def stop(self):
        """Stop the news ingestion engine"""
        if self.session:
            await self.session.close()
    
    async def _monitor_rss_feeds(self):
        """Monitor RSS feeds for new articles"""
        while True:
            try:
                for source, feeds in self.rss_feeds.items():
                    if source in self.config.sources:
                        for feed_url in feeds:
                            await self._process_rss_feed(source, feed_url)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring RSS feeds: {e}")
                await asyncio.sleep(60)
    
    async def _process_rss_feed(self, source: NewsSource, feed_url: str):
        """Process a single RSS feed"""
        try:
            async with self.session.get(feed_url) as response:
                if response.status == 200:
                    feed_content = await response.text()
                    feed = feedparser.parse(feed_content)
                    
                    for entry in feed.entries:
                        article_data = {
                            'title': entry.get('title', ''),
                            'url': entry.get('link', ''),
                            'published_at': self._parse_date(entry.get('published', '')),
                            'summary': entry.get('summary', ''),
                            'source': source
                        }
                        
                        # Check if we've already processed this article
                        article_hash = hashlib.md5(f"{article_data['url']}{article_data['title']}".encode()).hexdigest()
                        if article_hash not in self.processed_articles:
                            self.processed_articles.add(article_hash)
                            await self._fetch_full_article(article_data)
                            
        except Exception as e:
            logger.error(f"Error processing RSS feed {feed_url}: {e}")
    
    async def _fetch_full_article(self, article_data: Dict[str, Any]):
        """Fetch full article content"""
        try:
            article = Article(article_data['url'])
            article.download()
            article.parse()
            
            # Create NewsArticle object
            news_article = NewsArticle(
                id=hashlib.md5(f"{article_data['url']}{article_data['title']}".encode()).hexdigest(),
                title=article.title or article_data['title'],
                content=article.text,
                summary=article.summary or article_data['summary'],
                source=article_data['source'],
                url=article_data['url'],
                published_at=article_data['published_at'],
                processed_at=datetime.now(timezone.utc),
                author=', '.join(article.authors) if article.authors else None,
                word_count=len(article.text.split()) if article.text else 0
            )
            
            news_article.reading_time_minutes = news_article.word_count / 200  # Average reading speed
            
            # Queue for further processing
            await self._queue_article_for_processing(news_article)
            
        except Exception as e:
            logger.error(f"Error fetching full article {article_data['url']}: {e}")
    
    async def _poll_api_endpoints(self):
        """Poll premium API endpoints"""
        while True:
            try:
                for source, config in self.api_endpoints.items():
                    if source in self.config.sources and self.config.api_keys.get(source.value):
                        await self._fetch_from_api(source, config)
                
                await asyncio.sleep(600)  # Poll every 10 minutes
                
            except Exception as e:
                logger.error(f"Error polling API endpoints: {e}")
                await asyncio.sleep(300)
    
    async def _fetch_from_api(self, source: NewsSource, config: Dict[str, Any]):
        """Fetch articles from premium API"""
        try:
            # Rate limiting check
            now = time.time()
            recent_requests = [t for t in self.rate_limiters[source] if now - t < 3600]
            if len(recent_requests) >= self.config.max_articles_per_hour:
                return
            
            self.rate_limiters[source].append(now)
            
            # Build API request
            params = {
                'limit': 100,
                'since': (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            }
            
            if self.config.symbols_to_track:
                params['symbols'] = ','.join(self.config.symbols_to_track)
            
            async with self.session.get(
                config['base_url'],
                headers=config['headers'],
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    await self._process_api_response(source, data)
                    
        except Exception as e:
            logger.error(f"Error fetching from {source.value} API: {e}")
    
    async def _process_api_response(self, source: NewsSource, data: Dict[str, Any]):
        """Process API response and extract articles"""
        articles = data.get('articles', [])
        
        for article_data in articles:
            try:
                news_article = NewsArticle(
                    id=article_data.get('id', hashlib.md5(article_data.get('url', '').encode()).hexdigest()),
                    title=article_data.get('title', ''),
                    content=article_data.get('content', ''),
                    summary=article_data.get('summary', ''),
                    source=source,
                    url=article_data.get('url', ''),
                    published_at=self._parse_date(article_data.get('published_at', '')),
                    processed_at=datetime.now(timezone.utc),
                    author=article_data.get('author'),
                    symbols_mentioned=set(article_data.get('symbols', [])),
                    metadata=article_data.get('metadata', {})
                )
                
                if news_article.content:
                    news_article.word_count = len(news_article.content.split())
                    news_article.reading_time_minutes = news_article.word_count / 200
                
                await self._queue_article_for_processing(news_article)
                
            except Exception as e:
                logger.error(f"Error processing article from {source.value}: {e}")
    
    async def _queue_article_for_processing(self, article: NewsArticle):
        """Queue article for sentiment analysis and signal generation"""
        # This would typically send to a message queue (Redis, RabbitMQ, etc.)
        # For now, we'll emit an event
        logger.info(f"Queued article for processing: {article.title[:50]}...")
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        try:
            # Try common date formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # If all fail, return current time
            return datetime.now(timezone.utc)
        except:
            return datetime.now(timezone.utc)

class SentimentAnalysisEngine:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        self.models = {}
        self.nlp = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            # VADER sentiment analyzer
            if SentimentModel.VADER in self.config.sentiment_models:
                self.models['vader'] = SentimentIntensityAnalyzer()
            
            # FinBERT for financial sentiment
            if SentimentModel.FINBERT in self.config.sentiment_models:
                self.models['finbert'] = pipeline(
                    "sentiment-analysis",
                    model="yiyanghkust/finbert-tone",
                    tokenizer="yiyanghkust/finbert-tone"
                )
            
            # RoBERTa for general sentiment
            if SentimentModel.ROBERTA in self.config.sentiment_models:
                self.models['roberta'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            
            # spaCy for entity extraction
            if self.config.enable_entity_extraction:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model 'en_core_web_sm' not found. Entity extraction disabled.")
                    
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    async def analyze_article(self, article: NewsArticle) -> NewsArticle:
        """Perform comprehensive sentiment analysis on an article"""
        try:
            # Extract entities and symbols
            if self.config.enable_entity_extraction and self.nlp:
                article = await self._extract_entities(article)
            
            # Perform sentiment analysis with multiple models
            article = await self._analyze_sentiment(article)
            
            # Calculate relevance score
            article.relevance_score = self._calculate_relevance_score(article)
            
            # Calculate impact score
            article.impact_score = self._calculate_impact_score(article)
            
            # Categorize article
            article.categories = self._categorize_article(article)
            
            return article
            
        except Exception as e:
            logger.error(f"Error analyzing article {article.id}: {e}")
            return article
    
    async def _extract_entities(self, article: NewsArticle) -> NewsArticle:
        """Extract entities and financial symbols from article"""
        try:
            # Process with spaCy
            doc = self.nlp(article.content)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'GPE']:  # Organizations, persons, locations
                    article.entities.append(ent.text)
            
            # Extract potential stock symbols (pattern matching)
            symbol_pattern = r'\b[A-Z]{1,5}\b'
            potential_symbols = re.findall(symbol_pattern, article.content)
            
            # Validate symbols against known symbol list
            for symbol in potential_symbols:
                if symbol in self.config.symbols_to_track or len(symbol) <= 4:
                    article.symbols_mentioned.add(symbol)
            
            # Extract symbols from title as well
            title_symbols = re.findall(symbol_pattern, article.title)
            for symbol in title_symbols:
                if symbol in self.config.symbols_to_track or len(symbol) <= 4:
                    article.symbols_mentioned.add(symbol)
            
            return article
            
        except Exception as e:
            logger.error(f"Error extracting entities from article {article.id}: {e}")
            return article
    
    async def _analyze_sentiment(self, article: NewsArticle) -> NewsArticle:
        """Analyze sentiment using multiple models"""
        text_to_analyze = f"{article.title} {article.content}"
        
        # Limit text length for transformer models
        if len(text_to_analyze) > 10000:
            text_to_analyze = text_to_analyze[:10000]
        
        try:
            # VADER sentiment
            if 'vader' in self.models:
                vader_scores = self.models['vader'].polarity_scores(text_to_analyze)
                article.sentiment_scores['vader'] = {
                    'compound': vader_scores['compound'],
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu']
                }
            
            # TextBlob sentiment
            if SentimentModel.TEXTBLOB in self.config.sentiment_models:
                blob = TextBlob(text_to_analyze)
                article.sentiment_scores['textblob'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            
            # FinBERT sentiment
            if 'finbert' in self.models:
                try:
                    # Split text into chunks for FinBERT
                    chunks = [text_to_analyze[i:i+512] for i in range(0, len(text_to_analyze), 512)]
                    finbert_results = []
                    
                    for chunk in chunks[:5]:  # Limit to first 5 chunks
                        result = self.models['finbert'](chunk)
                        finbert_results.extend(result)
                    
                    # Aggregate results
                    if finbert_results:
                        avg_score = sum(r['score'] for r in finbert_results) / len(finbert_results)
                        dominant_label = max(set(r['label'] for r in finbert_results), 
                                           key=lambda x: sum(1 for r in finbert_results if r['label'] == x))
                        
                        article.sentiment_scores['finbert'] = {
                            'label': dominant_label,
                            'score': avg_score
                        }
                except Exception as e:
                    logger.error(f"FinBERT analysis failed: {e}")
            
            # RoBERTa sentiment
            if 'roberta' in self.models:
                try:
                    roberta_result = self.models['roberta'](text_to_analyze[:512])  # RoBERTa max length
                    if roberta_result:
                        article.sentiment_scores['roberta'] = {
                            'label': roberta_result[0]['label'],
                            'score': roberta_result[0]['score']
                        }
                except Exception as e:
                    logger.error(f"RoBERTa analysis failed: {e}")
            
            return article
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for article {article.id}: {e}")
            return article
    
    def _calculate_relevance_score(self, article: NewsArticle) -> float:
        """Calculate relevance score based on symbols, keywords, and content"""
        score = 0.0
        
        # Symbol mentions
        symbol_matches = len(article.symbols_mentioned.intersection(self.config.symbols_to_track))
        score += symbol_matches * 0.3
        
        # Keyword matches
        content_lower = article.content.lower()
        keyword_matches = sum(1 for keyword in self.config.keywords_to_track 
                            if keyword.lower() in content_lower)
        score += keyword_matches * 0.2
        
        # Title relevance
        title_lower = article.title.lower()
        title_matches = sum(1 for keyword in self.config.keywords_to_track 
                           if keyword.lower() in title_lower)
        score += title_matches * 0.4
        
        # Source credibility boost
        credible_sources = [NewsSource.REUTERS, NewsSource.BLOOMBERG, NewsSource.WSJ]
        if article.source in credible_sources:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_impact_score(self, article: NewsArticle) -> float:
        """Calculate potential market impact score"""
        impact = 0.0
        
        # Sentiment strength
        if 'vader' in article.sentiment_scores:
            impact += abs(article.sentiment_scores['vader']['compound']) * 0.3
        
        # Article recency (more recent = higher impact)
        hours_old = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
        recency_factor = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
        impact += recency_factor * 0.2
        
        # Source authority
        high_impact_sources = [NewsSource.REUTERS, NewsSource.BLOOMBERG, NewsSource.WSJ]
        if article.source in high_impact_sources:
            impact += 0.2
        
        # Content indicators
        content_lower = article.content.lower()
        impact_keywords = ['earnings', 'merger', 'acquisition', 'bankruptcy', 'lawsuit', 
                          'fda approval', 'analyst upgrade', 'analyst downgrade']
        
        for keyword in impact_keywords:
            if keyword in content_lower:
                impact += 0.1
        
        return min(impact, 1.0)
    
    def _categorize_article(self, article: NewsArticle) -> List[NewsCategory]:
        """Categorize article based on content"""
        categories = []
        content_title = f"{article.title} {article.content}".lower()
        
        # Earnings related
        if any(term in content_title for term in ['earnings', 'quarterly results', 'revenue', 'profit']):
            categories.append(NewsCategory.EARNINGS)
        
        # M&A related
        if any(term in content_title for term in ['merger', 'acquisition', 'takeover', 'buyout']):
            categories.append(NewsCategory.MERGER_ACQUISITION)
        
        # Analyst ratings
        if any(term in content_title for term in ['upgrade', 'downgrade', 'rating', 'price target']):
            categories.append(NewsCategory.ANALYST_RATING)
        
        # Regulatory
        if any(term in content_title for term in ['sec', 'fda', 'regulatory', 'compliance']):
            categories.append(NewsCategory.REGULATORY)
        
        # Economic data
        if any(term in content_title for term in ['gdp', 'inflation', 'employment', 'fed', 'interest rate']):
            categories.append(NewsCategory.ECONOMIC_DATA)
        
        # Default to general market if no specific category
        if not categories:
            categories.append(NewsCategory.GENERAL_MARKET)
        
        return categories

class SignalGenerationEngine:
    """Generate trading signals from news sentiment"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        self.article_buffer = defaultdict(list)  # Symbol -> [articles]
        self.signal_history = deque(maxlen=10000)
        self.symbol_sentiment_history = defaultdict(lambda: deque(maxlen=100))
    
    async def process_article(self, article: NewsArticle) -> List[NewsSignal]:
        """Process a single article and potentially generate signals"""
        signals = []
        
        # Only process articles above relevance threshold
        if article.relevance_score < self.config.relevance_threshold:
            return signals
        
        # Buffer article for each mentioned symbol
        for symbol in article.symbols_mentioned:
            if symbol in self.config.symbols_to_track:
                self.article_buffer[symbol].append(article)
                
                # Maintain buffer size
                if len(self.article_buffer[symbol]) > 50:
                    self.article_buffer[symbol] = self.article_buffer[symbol][-50:]
                
                # Generate signal for this symbol
                signal = await self._generate_signal_for_symbol(symbol, article)
                if signal:
                    signals.append(signal)
        
        return signals
    
    async def _generate_signal_for_symbol(self, symbol: str, article: NewsArticle) -> Optional[NewsSignal]:
        """Generate a trading signal for a specific symbol"""
        try:
            # Get recent articles for this symbol
            recent_articles = self.article_buffer[symbol][-10:]  # Last 10 articles
            
            # Calculate aggregate sentiment
            sentiment_scores = []
            total_impact = 0.0
            
            for art in recent_articles:
                if 'vader' in art.sentiment_scores:
                    sentiment_scores.append(art.sentiment_scores['vader']['compound'])
                elif 'finbert' in art.sentiment_scores:
                    # Convert FinBERT labels to scores
                    finbert_score = art.sentiment_scores['finbert']['score']
                    if art.sentiment_scores['finbert']['label'] == 'negative':
                        finbert_score = -finbert_score
                    elif art.sentiment_scores['finbert']['label'] == 'neutral':
                        finbert_score = 0
                    sentiment_scores.append(finbert_score)
                
                total_impact += art.impact_score
            
            if not sentiment_scores:
                return None
            
            # Calculate weighted sentiment
            weights = [art.impact_score for art in recent_articles]
            if sum(weights) == 0:
                return None
            
            weighted_sentiment = sum(s * w for s, w in zip(sentiment_scores, weights)) / sum(weights)
            
            # Check if sentiment is strong enough to generate a signal
            if abs(weighted_sentiment) < self.config.sentiment_threshold:
                return None
            
            # Determine signal type and strength
            signal_strength = weighted_sentiment
            signal_type = "momentum" if abs(weighted_sentiment) > 0.5 else "mean_reversion"
            
            # Calculate confidence based on consistency and impact
            sentiment_consistency = 1 - (np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0)
            confidence = min(sentiment_consistency * total_impact / len(recent_articles), 1.0)
            
            # Estimate time horizon based on news urgency
            avg_recency = sum((datetime.now(timezone.utc) - art.published_at).total_seconds() / 3600 
                            for art in recent_articles) / len(recent_articles)
            time_horizon = max(1, min(24, int(12 - avg_recency)))  # 1-24 hours
            
            # Create signal
            signal = NewsSignal(
                signal_id=f"news_{symbol}_{int(datetime.now(timezone.utc).timestamp())}",
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence=confidence,
                news_articles=[art.id for art in recent_articles],
                sentiment_aggregate=weighted_sentiment,
                volume_impact_expected=min(abs(weighted_sentiment) * 2, 1.0),
                price_impact_expected=abs(weighted_sentiment) * 0.02,  # Up to 2% price move
                time_horizon_hours=time_horizon,
                risk_level="high" if abs(weighted_sentiment) > 0.7 else "medium" if abs(weighted_sentiment) > 0.3 else "low",
                metadata={
                    'num_articles': len(recent_articles),
                    'avg_impact_score': total_impact / len(recent_articles),
                    'sentiment_consistency': sentiment_consistency
                }
            )
            
            # Store in history
            self.signal_history.append(signal)
            self.symbol_sentiment_history[symbol].append({
                'timestamp': datetime.now(timezone.utc),
                'sentiment': weighted_sentiment,
                'confidence': confidence
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for symbol {symbol}: {e}")
            return None
    
    def get_symbol_sentiment_trend(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get sentiment trend for a symbol over time"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        relevant_data = [
            data for data in self.symbol_sentiment_history[symbol]
            if data['timestamp'] > cutoff_time
        ]
        
        if not relevant_data:
            return {'trend': 'neutral', 'strength': 0.0, 'data_points': 0}
        
        sentiments = [data['sentiment'] for data in relevant_data]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Calculate trend
        if len(sentiments) > 1:
            recent_avg = sum(sentiments[-3:]) / min(3, len(sentiments))
            older_avg = sum(sentiments[:3]) / min(3, len(sentiments))
            trend_direction = recent_avg - older_avg
        else:
            trend_direction = 0
        
        trend = 'positive' if trend_direction > 0.1 else 'negative' if trend_direction < -0.1 else 'neutral'
        
        return {
            'trend': trend,
            'strength': abs(avg_sentiment),
            'direction': avg_sentiment,
            'data_points': len(relevant_data),
            'trend_strength': abs(trend_direction)
        }

class NewsAlphaEngine:
    """Main engine combining all components"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        self.ingestion_engine = NewsIngestionEngine(config)
        self.sentiment_engine = SentimentAnalysisEngine(config)
        self.signal_engine = SignalGenerationEngine(config)
        self.processed_articles = []
        self.generated_signals = []
        self.is_running = False
    
    async def start(self):
        """Start the news alpha engine"""
        self.is_running = True
        
        # Start ingestion
        await self.ingestion_engine.start()
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
        
        logger.info("News Alpha Engine started")
    
    async def stop(self):
        """Stop the news alpha engine"""
        self.is_running = False
        await self.ingestion_engine.stop()
        
        logger.info("News Alpha Engine stopped")
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # This would typically read from a message queue
                # For demo purposes, we'll process articles periodically
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(60)
    
    async def process_article(self, article: NewsArticle) -> Tuple[NewsArticle, List[NewsSignal]]:
        """Process a single article through the entire pipeline"""
        try:
            # Sentiment analysis
            analyzed_article = await self.sentiment_engine.analyze_article(article)
            
            # Signal generation
            signals = await self.signal_engine.process_article(analyzed_article)
            
            # Store results
            self.processed_articles.append(analyzed_article)
            self.generated_signals.extend(signals)
            
            # Maintain history size
            if len(self.processed_articles) > 10000:
                self.processed_articles = self.processed_articles[-5000:]
            
            return analyzed_article, signals
            
        except Exception as e:
            logger.error(f"Error processing article {article.id}: {e}")
            return article, []
    
    def get_latest_signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[NewsSignal]:
        """Get latest signals, optionally filtered by symbol"""
        signals = self.generated_signals[-limit:] if not symbol else [
            s for s in self.generated_signals[-limit*3:]  # Get more to filter
            if s.symbol == symbol
        ][-limit:]
        
        return sorted(signals, key=lambda x: x.timestamp, reverse=True)
    
    def get_symbol_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive summary for a symbol"""
        # Recent articles
        recent_articles = [
            art for art in self.processed_articles[-100:]
            if symbol in art.symbols_mentioned
        ]
        
        # Recent signals
        recent_signals = [
            sig for sig in self.generated_signals[-50:]
            if sig.symbol == symbol
        ]
        
        # Sentiment trend
        sentiment_trend = self.signal_engine.get_symbol_sentiment_trend(symbol)
        
        return {
            'symbol': symbol,
            'recent_articles_count': len(recent_articles),
            'recent_signals_count': len(recent_signals),
            'latest_sentiment': sentiment_trend,
            'last_article_time': max([art.published_at for art in recent_articles]) if recent_articles else None,
            'last_signal_time': max([sig.timestamp for sig in recent_signals]) if recent_signals else None,
            'avg_signal_strength': sum([abs(sig.signal_strength) for sig in recent_signals]) / len(recent_signals) if recent_signals else 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'total_articles_processed': len(self.processed_articles),
            'total_signals_generated': len(self.generated_signals),
            'symbols_tracked': len(self.config.symbols_to_track),
            'sources_active': len(self.config.sources),
            'models_active': len(self.config.sentiment_models),
            'is_running': self.is_running,
            'articles_last_hour': len([
                art for art in self.processed_articles
                if (datetime.now(timezone.utc) - art.processed_at).total_seconds() < 3600
            ]),
            'signals_last_hour': len([
                sig for sig in self.generated_signals
                if (datetime.now(timezone.utc) - sig.timestamp).total_seconds() < 3600
            ])
        }

# Example usage
async def example_usage():
    """Example of how to use the News Alpha Engine"""
    
    # Configure the engine
    config = NewsConfig(
        sources=[NewsSource.REUTERS, NewsSource.BLOOMBERG],
        sentiment_models=[SentimentModel.FINBERT, SentimentModel.VADER],
        symbols_to_track={'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'},
        keywords_to_track={'earnings', 'merger', 'acquisition', 'revenue'},
        max_articles_per_hour=500,
        sentiment_threshold=0.3,
        api_keys={
            'bloomberg': 'your_bloomberg_api_key',
            'reuters': 'your_reuters_api_key'
        }
    )
    
    # Create and start the engine
    engine = NewsAlphaEngine(config)
    await engine.start()
    
    try:
        # Let it run for a while
        await asyncio.sleep(3600)  # 1 hour
        
        # Get results
        signals = engine.get_latest_signals(limit=10)
        print(f"Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"Signal: {signal.symbol} - {signal.signal_type} - Strength: {signal.signal_strength:.2f}")
        
        # Get symbol summary
        aapl_summary = engine.get_symbol_summary('AAPL')
        print(f"AAPL Summary: {aapl_summary}")
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"Engine Stats: {stats}")
        
    finally:
        await engine.stop()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
