import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
import json

from .base_provider import BaseDataProvider, DataProviderError
from src.data.models import CompanyNews

logger = logging.getLogger(__name__)


class IndianNewsProvider:
    """Aggregator for Indian financial news from multiple sources."""
    
    def __init__(self):
        self.name = "Indian News Aggregator"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0  # 2 seconds between requests
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # News sources configuration
        self.news_sources = {
            'moneycontrol': {
                'base_url': 'https://www.moneycontrol.com',
                'search_url': 'https://www.moneycontrol.com/news/tags/{company}.html',
                'enabled': True
            },
            'economic_times': {
                'base_url': 'https://economictimes.indiatimes.com',
                'search_url': 'https://economictimes.indiatimes.com/topic/{company}',
                'enabled': True
            },
            'business_standard': {
                'base_url': 'https://www.business-standard.com',
                'search_url': 'https://www.business-standard.com/topic/{company}',
                'enabled': True
            },
            'livemint': {
                'base_url': 'https://www.livemint.com',
                'search_url': 'https://www.livemint.com/topic/{company}',
                'enabled': True
            }
        }
        
        # Company name mapping for Indian stocks
        self.company_mapping = {
            'RELIANCE': 'reliance-industries',
            'TCS': 'tata-consultancy-services',
            'INFY': 'infosys',
            'HDFCBANK': 'hdfc-bank',
            'ICICIBANK': 'icici-bank',
            'HINDUNILVR': 'hindustan-unilever',
            'ITC': 'itc',
            'SBIN': 'state-bank-of-india',
            'BHARTIARTL': 'bharti-airtel',
            'AXISBANK': 'axis-bank',
            'BAJFINANCE': 'bajaj-finance',
            'ASIANPAINT': 'asian-paints',
            'MARUTI': 'maruti-suzuki',
            'LT': 'larsen-toubro',
            'HCLTECH': 'hcl-technologies',
            'WIPRO': 'wipro',
            'TECHM': 'tech-mahindra',
            'ULTRACEMCO': 'ultratech-cement',
            'SUNPHARMA': 'sun-pharmaceutical',
            'POWERGRID': 'power-grid-corporation'
        }
    
    def _rate_limit(self):
        """Implement rate limiting to be respectful to news websites."""
        time.sleep(self.rate_limit_delay)
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol for news search."""
        ticker = ticker.upper()
        if ticker.endswith('.NS'):
            ticker = ticker[:-3]
        return ticker
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for news search from ticker."""
        normalized_ticker = self._normalize_ticker(ticker)
        return self.company_mapping.get(normalized_ticker, normalized_ticker.lower())
    
    def _make_request(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Make HTTP request with error handling."""
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            return None
    
    def _scrape_moneycontrol_news(self, ticker: str, limit: int = 10) -> List[CompanyNews]:
        """Scrape news from MoneyControl."""
        try:
            company_name = self._get_company_name(ticker)
            url = self.news_sources['moneycontrol']['search_url'].format(company=company_name)
            
            response = self._make_request(url)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news articles (MoneyControl specific selectors)
            articles = soup.find_all('div', class_='news_title')[:limit]
            
            for article in articles:
                try:
                    title_elem = article.find('a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Make URL absolute if relative
                    if url.startswith('/'):
                        url = self.news_sources['moneycontrol']['base_url'] + url
                    
                    # Try to get date and summary
                    date = datetime.now().strftime('%Y-%m-%d')
                    summary = title  # Use title as summary for now
                    
                    news_item = CompanyNews(
                        ticker=ticker,
                        title=title,
                        summary=summary,
                        url=url,
                        date=date,
                        source='MoneyControl',
                        sentiment=None
                    )
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"Error parsing MoneyControl article: {str(e)}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping MoneyControl news for {ticker}: {str(e)}")
            return []
    
    def _scrape_economic_times_news(self, ticker: str, limit: int = 10) -> List[CompanyNews]:
        """Scrape news from Economic Times."""
        try:
            company_name = self._get_company_name(ticker)
            url = self.news_sources['economic_times']['search_url'].format(company=company_name)
            
            response = self._make_request(url)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news articles (Economic Times specific selectors)
            articles = soup.find_all('div', class_='eachStory')[:limit]
            
            for article in articles:
                try:
                    title_elem = article.find('h3')
                    if not title_elem:
                        continue
                    
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        continue
                    
                    title = link_elem.get_text(strip=True)
                    url = link_elem.get('href', '')
                    
                    # Make URL absolute if relative
                    if url.startswith('/'):
                        url = self.news_sources['economic_times']['base_url'] + url
                    
                    # Try to get summary
                    summary_elem = article.find('p')
                    summary = summary_elem.get_text(strip=True) if summary_elem else title
                    
                    # Try to get date
                    date_elem = article.find('time')
                    date = date_elem.get('datetime', datetime.now().strftime('%Y-%m-%d')) if date_elem else datetime.now().strftime('%Y-%m-%d')
                    
                    news_item = CompanyNews(
                        ticker=ticker,
                        title=title,
                        summary=summary,
                        url=url,
                        date=date,
                        source='Economic Times',
                        sentiment=None
                    )
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"Error parsing Economic Times article: {str(e)}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping Economic Times news for {ticker}: {str(e)}")
            return []
    
    def _scrape_business_standard_news(self, ticker: str, limit: int = 10) -> List[CompanyNews]:
        """Scrape news from Business Standard."""
        try:
            company_name = self._get_company_name(ticker)
            url = self.news_sources['business_standard']['search_url'].format(company=company_name)
            
            response = self._make_request(url)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news articles (Business Standard specific selectors)
            articles = soup.find_all('article')[:limit]
            
            for article in articles:
                try:
                    title_elem = article.find('h2') or article.find('h3')
                    if not title_elem:
                        continue
                    
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        continue
                    
                    title = link_elem.get_text(strip=True)
                    url = link_elem.get('href', '')
                    
                    # Make URL absolute if relative
                    if url.startswith('/'):
                        url = self.news_sources['business_standard']['base_url'] + url
                    
                    # Try to get summary
                    summary_elem = article.find('p')
                    summary = summary_elem.get_text(strip=True) if summary_elem else title
                    
                    date = datetime.now().strftime('%Y-%m-%d')
                    
                    news_item = CompanyNews(
                        ticker=ticker,
                        title=title,
                        summary=summary,
                        url=url,
                        date=date,
                        source='Business Standard',
                        sentiment=None
                    )
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"Error parsing Business Standard article: {str(e)}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error scraping Business Standard news for {ticker}: {str(e)}")
            return []
    
    def get_aggregated_news(self, ticker: str, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None, limit: int = 50) -> List[CompanyNews]:
        """Get aggregated news from all Indian financial news sources."""
        all_news = []
        
        # Limit per source to ensure variety
        limit_per_source = max(5, limit // len([s for s in self.news_sources.values() if s['enabled']]))
        
        # MoneyControl
        if self.news_sources['moneycontrol']['enabled']:
            try:
                mc_news = self._scrape_moneycontrol_news(ticker, limit_per_source)
                all_news.extend(mc_news)
                logger.info(f"Fetched {len(mc_news)} articles from MoneyControl for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching MoneyControl news: {str(e)}")
        
        # Economic Times
        if self.news_sources['economic_times']['enabled']:
            try:
                et_news = self._scrape_economic_times_news(ticker, limit_per_source)
                all_news.extend(et_news)
                logger.info(f"Fetched {len(et_news)} articles from Economic Times for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching Economic Times news: {str(e)}")
        
        # Business Standard
        if self.news_sources['business_standard']['enabled']:
            try:
                bs_news = self._scrape_business_standard_news(ticker, limit_per_source)
                all_news.extend(bs_news)
                logger.info(f"Fetched {len(bs_news)} articles from Business Standard for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching Business Standard news: {str(e)}")
        
        # Remove duplicates based on title similarity
        unique_news = []
        seen_titles = set()
        
        for news_item in all_news:
            # Create a simplified title for comparison
            simple_title = news_item.title.lower().replace(' ', '').replace('-', '')[:50]
            if simple_title not in seen_titles:
                seen_titles.add(simple_title)
                unique_news.append(news_item)
        
        # Sort by date (newest first) and limit results
        try:
            unique_news.sort(key=lambda x: x.date, reverse=True)
        except:
            pass  # If date parsing fails, keep original order
        
        return unique_news[:limit]
    
    def get_market_sentiment(self, ticker: str = None) -> Dict[str, Any]:
        """Get overall Indian market sentiment from news analysis."""
        try:
            # If ticker is provided, get sentiment for that stock
            if ticker:
                news = self.get_aggregated_news(ticker, limit=20)
            else:
                # Get general market news from Nifty or overall market
                news = self.get_aggregated_news('NIFTY', limit=50)
            
            if not news:
                return {'sentiment': 'neutral', 'confidence': 0.0, 'news_count': 0}
            
            # Simple sentiment analysis based on keywords
            positive_keywords = ['gain', 'rise', 'up', 'profit', 'growth', 'bullish', 'positive', 'strong', 'boost', 'surge']
            negative_keywords = ['fall', 'down', 'loss', 'drop', 'decline', 'bearish', 'negative', 'weak', 'crash', 'slump']
            
            positive_count = 0
            negative_count = 0
            
            for article in news:
                title_lower = article.title.lower()
                summary_lower = article.summary.lower() if article.summary else ''
                text = title_lower + ' ' + summary_lower
                
                for keyword in positive_keywords:
                    if keyword in text:
                        positive_count += 1
                        break
                
                for keyword in negative_keywords:
                    if keyword in text:
                        negative_count += 1
                        break
            
            total_classified = positive_count + negative_count
            if total_classified == 0:
                sentiment = 'neutral'
                confidence = 0.0
            else:
                confidence = total_classified / len(news)
                if positive_count > negative_count:
                    sentiment = 'positive'
                elif negative_count > positive_count:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'total_articles': len(news),
                'classified_articles': total_classified
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'news_count': 0}