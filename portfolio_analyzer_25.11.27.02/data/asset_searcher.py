import requests
import time
from typing import List, Optional, Dict, Any
import logging

from data.asset_info import AssetInfo


class AssetSearcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.search_url = "https://query1.finance.yahoo.com/v1/finance/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        self.last_request_time = 0
        self.min_request_interval = 0.2
    
    def search_assets(self, query: str, max_results: int = 20) -> List[AssetInfo]:
        if not query or len(query.strip()) < 1:
            return []
        
        try:
            self._rate_limit()
            search_results = self._call_yahoo_search_api(query.strip())
            
            if not search_results:
                return []
            
            assets = []
            for result in search_results[:max_results]:
                asset = self._convert_to_asset_info(result)
                if asset:
                    assets.append(asset)
            
            return assets
            
        except Exception as e:
            self.logger.error(f"Search error for '{query}': {e}")
            return []
    
    def _rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def _call_yahoo_search_api(self, query: str) -> List[Dict[str, Any]]:
        try:
            params = {'q': query, 'quotesCount': 15, 'newsCount': 0}
            response = self.session.get(self.search_url, params=params, timeout=3)
            response.raise_for_status()
            data = response.json()
            return data.get('quotes', [])
        except Exception as e:
            self.logger.error(f"API error: {e}")
            return []
    
    def _convert_to_asset_info(self, yahoo_result: Dict[str, Any]) -> Optional[AssetInfo]:
        try:
            symbol = yahoo_result.get('symbol')
            if not symbol:
                return None
            
            name = (yahoo_result.get('longname') or 
                   yahoo_result.get('shortname') or 
                   symbol)
            
            exchange = yahoo_result.get('exchange', '')
            yf_currency = yahoo_result.get('currency')
            inferred_currency = AssetInfo._static_infer_currency_from_symbol(symbol)
            
            currency = yf_currency
            if (symbol.startswith('^') or 
                symbol in AssetInfo._get_known_symbol_currencies() or
                not yf_currency or 
                (yf_currency == 'USD' and inferred_currency != 'USD')):
                currency = inferred_currency
            
            country = self._infer_country(exchange, symbol)
            
            return AssetInfo(
                symbol=symbol, name=name, exchange=exchange, currency=currency,
                country=country, sector=yahoo_result.get('sector'),
                industry=yahoo_result.get('industry'), 
                legal_type=yahoo_result.get('quoteType')
            )
            
        except Exception as e:
            self.logger.error(f"Conversion error: {e}")
            return None
    
    def _infer_country(self, exchange: str, symbol: str) -> str:
        exchange_country_map = {
            'JPX': 'Japan', 'TSE': 'Japan', 'TYO': 'Japan',
            'NASDAQ': 'United States', 'NYSE': 'United States', 'NYQ': 'United States',
            'OEM': 'United States', 'OTC': 'United States',
            'LSE': 'United Kingdom', 'LON': 'United Kingdom',
            'FRA': 'Germany', 'XETRA': 'Germany', 'PAR': 'France',
            'MIL': 'Italy', 'TSX': 'Canada', 'ASX': 'Australia',
            'HKEX': 'Hong Kong', 'HKG': 'Hong Kong',
        }
        
        if exchange.upper() in exchange_country_map:
            return exchange_country_map[exchange.upper()]
        
        symbol_upper = symbol.upper()
        
        if symbol_upper.startswith('^'):
            if any(jp in symbol_upper for jp in ['N225', 'TPX', 'NIKKEI']):
                return 'Japan'
            elif 'FTSE' in symbol_upper:
                return 'United Kingdom'
            elif any(de in symbol_upper for de in ['DAX', 'GDAX']):
                return 'Germany'
            elif 'FCHI' in symbol_upper or 'CAC' in symbol_upper:
                return 'France'
            elif 'HSI' in symbol_upper:
                return 'Hong Kong'
            elif 'AX' in symbol_upper:
                return 'Australia'
            else:
                return 'United States'
        
        suffix_country_map = {
            ('.T',): 'Japan', ('.L',): 'United Kingdom', ('.F', '.DE'): 'Germany',
            ('.PA',): 'France', ('.MI',): 'Italy', ('.TO',): 'Canada',
            ('.AX',): 'Australia', ('.HK',): 'Hong Kong', ('.SS', '.SZ'): 'China'
        }
        
        for suffixes, country in suffix_country_map.items():
            if any(symbol_upper.endswith(s) for s in suffixes):
                return country
        
        if symbol.isdigit() and len(symbol) == 4:
            return 'Japan'
        
        return 'United States'
    
    def validate_symbol(self, symbol: str) -> bool:
        try:
            results = self.search_assets(symbol, max_results=1)
            return len(results) > 0
        except:
            return False