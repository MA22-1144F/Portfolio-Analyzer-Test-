from dataclasses import dataclass
from typing import Optional, Dict, Any
import yfinance as yf


@dataclass
class AssetInfo:
    symbol: str
    name: str
    exchange: Optional[str] = None
    currency: Optional[str] = None
    country: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    legal_type: Optional[str] = None
    market_cap: Optional[float] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if not self.name:
            self.name = self.symbol
        
        if not self.currency:
            self.currency = self._infer_currency_from_symbol(self.symbol)
        if not self.country:
            self.country = self._infer_country_from_exchange_and_symbol(self.exchange, self.symbol)
        if not self.exchange:
            self.exchange = self._infer_exchange_from_symbol(self.symbol)
    
    def get_sector_or_type(self) -> str:
        return self.sector or self.legal_type or "-"
    
    @classmethod
    def from_yfinance_info(cls, symbol: str, yf_info: Dict[str, Any]) -> 'AssetInfo':
        name = (yf_info.get('longName') or 
                yf_info.get('shortName') or 
                yf_info.get('displayName') or 
                symbol)
        
        exchange = (yf_info.get('exchange') or 
                   yf_info.get('fullExchangeName'))
        
        currency = yf_info.get('currency')
        inferred_currency = cls._static_infer_currency_from_symbol(symbol)
        
        if (symbol.startswith('^') or 
            symbol in cls._get_known_symbol_currencies() or
            not currency or 
            currency == 'USD' and inferred_currency != 'USD'):
            currency = inferred_currency
        
        return cls(
            symbol=symbol,
            name=name,
            exchange=exchange,
            currency=currency,
            country=yf_info.get('country'),
            sector=yf_info.get('sector'),
            industry=yf_info.get('industry'),
            legal_type=yf_info.get('quoteType'),
            market_cap=yf_info.get('marketCap'),
            description=yf_info.get('longBusinessSummary')
        )
    
    @staticmethod
    def _get_known_symbol_currencies() -> Dict[str, str]:
        return {
            '^N225': 'JPY', '^TPX': 'JPY', '^GSPC': 'USD', '^DJI': 'USD',
            '^IXIC': 'USD', '^FTSE': 'GBP', '^GDAXI': 'EUR', '^FCHI': 'EUR',
            '^HSI': 'HKD', '^AXJO': 'AUD', 'USDJPY=X': 'JPY', 'EURUSD=X': 'USD',
            'BTC-USD': 'USD', 'ETH-USD': 'USD'
        }
    
    def _infer_currency_from_symbol(self, symbol: str) -> str:
        return self._static_infer_currency_from_symbol(symbol)
    
    @staticmethod
    def _static_infer_currency_from_symbol(symbol: str) -> str:
        symbol_upper = symbol.upper()
        
        known_currencies = AssetInfo._get_known_symbol_currencies()
        if symbol in known_currencies:
            return known_currencies[symbol]
        
        if symbol_upper.startswith('^'):
            if any(jp in symbol_upper for jp in ['N225', 'TPX', 'NIKKEI']):
                return 'JPY'
            elif 'FTSE' in symbol_upper:
                return 'GBP'
            elif any(de in symbol_upper for de in ['DAX', 'GDAX']):
                return 'EUR'
            elif 'FCHI' in symbol_upper or 'CAC' in symbol_upper:
                return 'EUR'
            elif 'HSI' in symbol_upper:
                return 'HKD'
            elif 'AX' in symbol_upper:
                return 'AUD'
            else:
                return 'USD'
        
        if symbol_upper.endswith('=X'):
            if symbol_upper.startswith('USD'):
                pair = symbol_upper.replace('USD', '').replace('=X', '')
                return pair if pair else 'USD'
            elif symbol_upper.endswith('USD=X'):
                return 'USD'
            else:
                return symbol_upper.replace('=X', '')[-3:]
        
        if symbol_upper.endswith('-USD'):
            return 'USD'
        
        suffix_map = {
            ('.T', '.JP'): 'JPY', '.L': 'GBP', ('.F', '.DE'): 'EUR',
            '.PA': 'EUR', '.MI': 'EUR', '.TO': 'CAD', '.AX': 'AUD',
            '.HK': 'HKD', ('.SS', '.SZ'): 'CNY'
        }
        
        for suffixes, currency in suffix_map.items():
            if isinstance(suffixes, tuple):
                if any(symbol_upper.endswith(s) for s in suffixes):
                    return currency
            elif symbol_upper.endswith(suffixes):
                return currency
        
        if symbol.isdigit() and len(symbol) == 4:
            return 'JPY'
        
        return 'USD'
    
    def _infer_country_from_exchange_and_symbol(self, exchange: Optional[str], symbol: str) -> str:
        exchange_country_map = {
            'JPX': 'Japan', 'TSE': 'Japan', 'TYO': 'Japan',
            'NASDAQ': 'United States', 'NYSE': 'United States', 'NYQ': 'United States',
            'LSE': 'United Kingdom', 'LON': 'United Kingdom',
            'FRA': 'Germany', 'XETRA': 'Germany', 'PAR': 'France',
            'MIL': 'Italy', 'TSX': 'Canada', 'TOR': 'Canada',
            'ASX': 'Australia', 'HKEX': 'Hong Kong', 'HKG': 'Hong Kong'
        }
        
        if exchange and exchange.upper() in exchange_country_map:
            return exchange_country_map[exchange.upper()]
        
        symbol_upper = symbol.upper()
        
        if symbol_upper.startswith('^'):
            if any(jp in symbol_upper for jp in ['N225', 'TPX', 'NIKKEI']):
                return 'Japan'
            elif 'FTSE' in symbol_upper:
                return 'United Kingdom'
            elif 'DAX' in symbol_upper:
                return 'Germany'
            elif 'CAC' in symbol_upper or 'FCHI' in symbol_upper:
                return 'France'
            elif 'HSI' in symbol_upper:
                return 'Hong Kong'
            elif 'AX' in symbol_upper:
                return 'Australia'
            else:
                return 'United States'
        
        suffix_country_map = {
            ('.T', '.JP'): 'Japan', '.L': 'United Kingdom', ('.F', '.DE'): 'Germany',
            '.PA': 'France', '.MI': 'Italy', '.TO': 'Canada', '.AX': 'Australia',
            '.HK': 'Hong Kong', ('.SS', '.SZ'): 'China'
        }
        
        for suffixes, country in suffix_country_map.items():
            if isinstance(suffixes, tuple):
                if any(symbol_upper.endswith(s) for s in suffixes):
                    return country
            elif symbol_upper.endswith(suffixes):
                return country
        
        if symbol.isdigit() and len(symbol) == 4:
            return 'Japan'
        
        return 'United States'
    
    def _infer_exchange_from_symbol(self, symbol: str) -> str:
        symbol_upper = symbol.upper()
        
        if symbol_upper.endswith('.T') or (symbol.isdigit() and len(symbol) == 4):
            return 'JPX'
        elif symbol_upper.endswith('.L'):
            return 'LSE'
        elif symbol_upper.endswith('.F'):
            return 'FRA'
        elif symbol_upper.endswith('.DE'):
            return 'XETRA'
        elif symbol_upper.endswith('.PA'):
            return 'PAR'
        elif symbol_upper.endswith('.MI'):
            return 'MIL'
        elif symbol_upper.endswith('.TO'):
            return 'TSX'
        elif symbol_upper.endswith('.AX'):
            return 'ASX'
        elif symbol_upper.endswith('.HK'):
            return 'HKEX'
        elif symbol_upper.startswith('^'):
            if any(jp in symbol_upper for jp in ['N225', 'TPX']):
                return 'JPX'
            elif 'FTSE' in symbol_upper:
                return 'LSE'
            elif 'DAX' in symbol_upper:
                return 'XETRA'
            elif 'CAC' in symbol_upper or 'FCHI' in symbol_upper:
                return 'PAR'
            elif 'HSI' in symbol_upper:
                return 'HKEX'
            elif 'AX' in symbol_upper:
                return 'ASX'
            else:
                return 'NASDAQ'
        else:
            return 'NASDAQ'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol, 'name': self.name, 'exchange': self.exchange,
            'currency': self.currency, 'country': self.country, 'sector': self.sector,
            'industry': self.industry, 'legal_type': self.legal_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssetInfo':
        return cls(
            symbol=data['symbol'], name=data['name'], exchange=data.get('exchange'),
            currency=data.get('currency'), country=data.get('country'), 
            sector=data.get('sector'), industry=data.get('industry'),
            legal_type=data.get('legal_type')
        )
    
    def is_valid_for_analysis(self) -> bool:
        if not self.symbol or not self.name:
            return False
        
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="1mo")
            return len(hist) > 0
        except:
            return False
    
    def __str__(self) -> str:
        return f"{self.name} ({self.symbol})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, AssetInfo) and self.symbol == other.symbol
    
    def __hash__(self) -> int:
        return hash(self.symbol)