import json
from pathlib import Path
from typing import Dict, Any, Optional


class AppConfig:
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.config_file = config_file or (self.config_dir / "user_settings.json")
        self.default_config_file = self.config_dir / "default_settings.json"
        
        self.config_dir.mkdir(exist_ok=True)
        self._settings: Dict[str, Any] = {}
        
        self.load_default_settings()
        self.load_user_settings()
    
    def load_default_settings(self):
        try:
            if self.default_config_file.exists():
                with open(self.default_config_file, 'r', encoding='utf-8') as f:
                    self._settings.update(json.load(f))
            else:
                self.create_default_config()
        except (json.JSONDecodeError, IOError):
            self.create_default_config()
    
    def create_default_config(self):
        default_settings = {
            "app": {
                "name": "Portfolio Analyzer",
                "version": "1.0.0"
            },
            "window": {
                "width": 1200,
                "height": 700,
                "last_tab_index": 0
            },
            "analysis": {
                "default_risk_free_rate": 0.0,
                "default_span": "日次",
                "default_period_days": 365,
                "min_data_points": {
                    "日次": 30,
                    "週次": 20,
                    "月次": 12,
                    "年次": 3
                },
                "min_coverage_ratio": 0.7,
                "time_conversion_factors": {
                    "business_days_per_year": 252,
                    "calendar_days_per_year": 365,
                    "weeks_per_year": 52,
                    "months_per_year": 12,
                    "span_factors": {
                        "日次": "calendar_days_per_year",
                        "週次": "weeks_per_year",
                        "月次": "months_per_year",
                        "年次": 1
                    }
                },
                "widget_sizes": {
                    "price_series": {"min_height": 600, "preferred_height": 650, "min_width": 800, "preferred_width": 900},
                    "return_risk_analysis": {"min_height": 800, "preferred_height": 850, "min_width": 700, "preferred_width": 800},
                    "correlation_matrix": {"min_height": 600, "preferred_height": 650, "min_width": 600, "preferred_width": 700},
                    "efficient_frontier": {"min_height": 600, "preferred_height": 700, "min_width": 800, "preferred_width": 900},
                    "security_market_line": {"min_height": 500, "preferred_height": 600, "min_width": 700, "preferred_width": 800}
                },
                "time_conversion": {
                    "日次": {
                        "to_weekly": 7,
                        "to_monthly": 30,
                        "to_yearly": 365
                    },
                    "週次": {
                        "to_daily": 0.142857,  # 1/7
                        "to_monthly": 4,
                        "to_yearly": 52
                    },
                    "月次": {
                        "to_daily": 0.033333,  # 1/30
                        "to_weekly": 0.25,     # 1/4
                        "to_yearly": 12
                    },
                    "年次": {
                        "to_daily": 0.002740,  # 1/365
                        "to_weekly": 0.019231, # 1/52
                        "to_monthly": 0.083333 # 1/12
                    }
                },
                "risk_conversion": {
                    "日次": {
                        "to_weekly": 2.645751,   # sqrt(7)
                        "to_monthly": 5.477226,  # sqrt(30)
                        "to_yearly": 19.104973   # sqrt(365)
                    },
                    "週次": {
                        "to_daily": 0.377964,   # 1/sqrt(7)
                        "to_monthly": 2.0,      # sqrt(4)
                        "to_yearly": 7.211103   # sqrt(52)
                    },
                    "月次": {
                        "to_daily": 0.182574,   # 1/sqrt(30)
                        "to_weekly": 0.5,       # 1/sqrt(4)
                        "to_yearly": 3.464102   # sqrt(12)
                    },
                    "年次": {
                        "to_daily": 0.052315,   # 1/sqrt(365)
                        "to_weekly": 0.138675,  # 1/sqrt(52)
                        "to_monthly": 0.288675  # 1/sqrt(12)
                    }
                }
            },
            "ui": {
                "colors": {
                    "primary": "#0078d4",
                    "primary_hover": "#106ebe",
                    "secondary": "#28a745",
                    "secondary_hover": "#218838",
                    "danger": "#dc3545",
                    "danger_hover": "#c82333",
                    "background": "#2b2b2b",
                    "surface": "#3c3c3c",
                    "border": "#555555",
                    "text_primary": "#ffffff",
                    "text_secondary": "#aaaaaa",
                    "text_accent": "#0078d4",
                    "grid_line": "#444444"
                },
                "sizes": {
                    "progress_bar_height": 18,
                    "status_label_height": 12,
                    "quality_info_height": 30,
                    "button_min_width": 60,
                    "header_button_height": 25,
                    "small_button_width": 80,
                    "medium_button_width": 120,
                    "large_button_width": 180,
                    "empty_state_min_height": 80,
                    "empty_state_max_height": 150,
                    "button_height": 15,
                    "button_font_size": 10,
                    "button_border_radius": 4
                },
                "button_types": {
                    "primary": {
                        "width": 100,
                        "bg_color": "#0078d4",
                        "hover_color": "#106ebe",
                        "text_color": "#ffffff"
                    },
                    "secondary": {
                        "width": 80,
                        "bg_color": "#7e13e3",
                        "hover_color": "#6201c4",
                        "text_color": "#ffffff"
                    },
                    "export": {
                        "width": 90,
                        "bg_color": "#28a745",
                        "hover_color": "#218838",
                        "text_color": "#ffffff"
                    },
                    "save": {
                        "width": 80,
                        "bg_color": "#1a8408",
                        "hover_color": "#106e1e",
                        "text_color": "#ffffff"
                    },
                    "danger": {
                        "width": 70,
                        "bg_color": "#dc3545",
                        "hover_color": "#c82333",
                        "text_color": "#ffffff"
                    },
                    "info": {
                        "width": 60,
                        "bg_color": "#6c757d",
                        "hover_color": "#5a6268",
                        "text_color": "#ffffff"
                    },
                    "small": {
                        "width": 50,
                        "bg_color": "#6f42c1",
                        "hover_color": "#5a32a3",
                        "text_color": "#ffffff"
                    },
                    "neutral": {
                        "width": 70,
                        "bg_color": "#6c757d",
                        "hover_color": "#545b62",
                        "text_color": "#ffffff"
                    }
                },
                "layout": {
                    "main_margins": [5, 5, 5, 5],
                    "main_spacing": 3,
                    "header_spacing": 8,
                    "content_spacing": 5
                },
                "analysis_widget_sizes": {
                    "price_series": {
                        "min_height": 600,
                        "preferred_height": 650,
                        "min_width": 800,
                        "preferred_width": 900
                    },
                    "return_risk_analysis": {
                        "min_height": 800,
                        "preferred_height": 850,
                        "min_width": 700,
                        "preferred_width": 800
                    },
                    "correlation_matrix": {
                        "min_height": 600,
                        "preferred_height": 650,
                        "min_width": 600,
                        "preferred_width": 700
                    },
                    "efficient_frontier": {
                        "min_height": 600,
                        "preferred_height": 700,
                        "min_width": 800,
                        "preferred_width": 900
                    },
                    "capital_market_line": {
                        "min_height": 500,
                        "preferred_height": 600,
                        "min_width": 700,
                        "preferred_width": 800
                    },
                    "security_market_line": {
                        "min_height": 500,
                        "preferred_height": 600,
                        "min_width": 700,
                        "preferred_width": 800
                    }
                },
                "default_analysis_size": {
                    "min_height": 400,
                    "preferred_height": 500,
                    "min_width": 600,
                    "preferred_width": 700
                }
            },
            "market": {
                "portfolios": {
                    "Nikkei 225 (^N225)": {
                        "ticker": "^N225",
                        "name": "Nikkei 225",
                        "country": "Japan",
                        "currency": "JPY",
                        "description": "日経平均株価"
                    },
                    "NASDAQ Composite (^IXIC)": {
                        "ticker": "^IXIC",
                        "name": "NASDAQ Composite",
                        "country": "United States",
                        "currency": "USD",
                        "description": "NASDAQ総合指数"
                    },
                    "S&P 500 (^GSPC)": {
                        "ticker": "^GSPC",
                        "name": "S&P 500",
                        "country": "United States",
                        "currency": "USD",
                        "description": "S&P500指数"
                    },
                    "Dow Jones Industrial Average (^DJI)": {
                        "ticker": "^DJI",
                        "name": "Dow Jones Industrial Average",
                        "country": "United States",
                        "currency": "USD",
                        "description": "ダウ工業株30種"
                    }
                }
            },
            "data": {
                "cache_enabled": True,
                "cache_duration_days": 1
            }
        }
        
        try:
            with open(self.default_config_file, 'w', encoding='utf-8') as f:
                json.dump(default_settings, f, indent=2, ensure_ascii=False)
            self._settings.update(default_settings)
        except IOError:
            pass
    
    def load_user_settings(self):
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)
                    self._deep_update(self._settings, user_settings)
        except (json.JSONDecodeError, IOError):
            pass
    
    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
        except IOError:
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        current = self._settings
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    # UI設定取得メソッド
    def get_ui_colors(self) -> Dict[str, str]:
        """UI色設定を取得"""
        return self.get('ui.colors', {})
    
    def get_ui_sizes(self) -> Dict[str, int]:
        """UIサイズ設定を取得"""
        return self.get('ui.sizes', {})
    
    def get_button_types(self) -> Dict[str, Dict[str, Any]]:
        """ボタンタイプ設定を取得"""
        return self.get('ui.button_types', {})
    
    def get_layout_settings(self) -> Dict[str, Any]:
        """レイアウト設定を取得"""
        return self.get('ui.layout', {})
    
    def get_analysis_widget_size(self, item_type: str) -> Dict[str, int]:
        """分析ウィジェットサイズを取得"""
        sizes = self.get('ui.analysis_widget_sizes', {})
        return sizes.get(item_type, self.get('ui.default_analysis_size', {}))
    
    # 分析設定取得メソッド
    def get_min_data_points(self, span: str) -> int:
        """最小データポイント数を取得"""
        min_points = self.get('analysis.min_data_points', {})
        return min_points.get(span, 30)
    
    def get_min_coverage_ratio(self) -> float:
        """最小カバレッジ率を取得"""
        return self.get('analysis.min_coverage_ratio', 0.7)
    
    def get_time_conversion_factor(self, from_span: str, to_span: str) -> float:
        """時間換算係数を取得"""
        conversions = self.get('analysis.time_conversion', {})
        from_conversions = conversions.get(from_span, {})
        key = f"to_{to_span.replace('次', 'ly').replace('日', 'dai').replace('週', 'week').replace('月', 'month').replace('年', 'year')}"
        return from_conversions.get(key, 1.0)
    
    def get_risk_conversion_factor(self, from_span: str, to_span: str) -> float:
        """リスク換算係数を取得"""
        conversions = self.get('analysis.risk_conversion', {})
        from_conversions = conversions.get(from_span, {})
        key = f"to_{to_span.replace('次', 'ly').replace('日', 'dai').replace('週', 'week').replace('月', 'month').replace('年', 'year')}"
        return from_conversions.get(key, 1.0)
    
    # 市場設定取得メソッド
    def get_market_portfolios(self) -> Dict[str, Dict[str, str]]:
        """市場ポートフォリオ設定を取得"""
        return self.get('market.portfolios', {})
    
    def get_market_info(self, market_text: str) -> Optional[Dict[str, str]]:
        """特定の市場情報を取得"""
        portfolios = self.get_market_portfolios()
        return portfolios.get(market_text)
    
    # 既存プロパティ
    @property
    def window_geometry(self):
        return self.get('window.geometry')
    
    @window_geometry.setter
    def window_geometry(self, value):
        self.set('window.geometry', value)
    
    @property
    def last_tab_index(self):
        return self.get('window.last_tab_index', 0)
    
    @last_tab_index.setter
    def last_tab_index(self, value):
        self.set('window.last_tab_index', value)
    
    # テーマ設定メソッド
    def apply_color_theme(self, theme_name: str):
        """カラーテーマを適用"""
        themes = {
            'dark': {
                "primary": "#0078d4",
                "primary_hover": "#106ebe",
                "secondary": "#28a745",
                "secondary_hover": "#218838",
                "danger": "#dc3545",
                "danger_hover": "#c82333",
                "background": "#2b2b2b",
                "surface": "#3c3c3c",
                "border": "#555555",
                "text_primary": "#ffffff",
                "text_secondary": "#aaaaaa",
                "text_accent": "#0078d4",
                "grid_line": "#444444"
            },
            'light': {
                "primary": "#0078d4",
                "primary_hover": "#106ebe",
                "secondary": "#28a745",
                "secondary_hover": "#218838",
                "danger": "#dc3545",
                "danger_hover": "#c82333",
                "background": "#ffffff",
                "surface": "#f8f9fa",
                "border": "#dee2e6",
                "text_primary": "#000000",
                "text_secondary": "#6c757d",
                "text_accent": "#0078d4",
                "grid_line": "#dee2e6"
            }
        }
        
        if theme_name in themes:
            self.set('ui.colors', themes[theme_name])
            self.save()
    
    def scale_ui_sizes(self, scale_factor: float):
        """UIサイズを一括スケーリング"""
        sizes = self.get_ui_sizes()
        scaled_sizes = {key: int(value * scale_factor) for key, value in sizes.items()}
        self.set('ui.sizes', scaled_sizes)
        
        # 分析ウィジェットサイズもスケーリング
        widget_sizes = self.get('ui.analysis_widget_sizes', {})
        for widget_type, size_config in widget_sizes.items():
            scaled_config = {key: int(value * scale_factor) for key, value in size_config.items()}
            self.set(f'ui.analysis_widget_sizes.{widget_type}', scaled_config)
        
        # デフォルトサイズもスケーリング
        default_size = self.get('ui.default_analysis_size', {})
        scaled_default = {key: int(value * scale_factor) for key, value in default_size.items()}
        self.set('ui.default_analysis_size', scaled_default)
        
        self.save()


# グローバル設定インスタンス
_config_instance = None

def get_config() -> AppConfig:
    """グローバル設定インスタンスを取得"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

def initialize_config(config_file: Optional[str] = None) -> AppConfig:
    """設定を初期化"""
    global _config_instance
    _config_instance = AppConfig(config_file)
    return _config_instance