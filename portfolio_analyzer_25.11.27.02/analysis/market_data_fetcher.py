# portfolio_analyzer/analysis/market_data_fetcher.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging
import time
from config.app_config import get_config


class MarketDataFetcher:
    """市場ポートフォリオデータ取得クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # 市場ポートフォリオの定義を設定から取得
        self.market_portfolios = self.config.get_market_portfolios()
    
    def get_market_info(self, market_text: str) -> Optional[Dict[str, str]]:
        """市場情報を取得"""
        return self.config.get_market_info(market_text)
    
    def extract_ticker_from_market_text(self, market_text: str) -> str:
        """市場テキストからティッカーを抽出"""
        market_info = self.get_market_info(market_text)
        if market_info:
            return market_info["ticker"]
        
        # フォールバック: 括弧内からティッカーを抽出
        try:
            if "(" in market_text and ")" in market_text:
                return market_text.split("(")[1].replace(")", "")
        except (IndexError, AttributeError):
            pass
        
        # デフォルト
        return "^N225"
    
    def _adjust_date_range_for_span(self, start_date: datetime, end_date: datetime, span: str) -> Tuple[datetime, datetime]:
        """スパンに応じて期間を調整"""
        if span == '週次':
            # 週次データには最低3ヶ月の期間を確保
            min_period = timedelta(days=90)
            if (end_date - start_date) < min_period:
                start_date = end_date - min_period
                self.logger.info(f"週次データのため期間を調整: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        elif span == '月次':
            # 月次データには最低2年の期間を確保
            min_period = timedelta(days=730)
            if (end_date - start_date) < min_period:
                start_date = end_date - min_period
                self.logger.info(f"月次データのため期間を調整: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        
        return start_date, end_date
    
    def fetch_market_data(self, market_text: str, start_date: datetime, 
                         end_date: datetime, span: str) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
        """
        市場ポートフォリオデータを取得（Stock Splits エラー対応版）
        
        Returns:
            Tuple[Optional[pd.Series], Dict[str, Any]]: (市場リターンデータ, ステータス情報)
        """
        market_info = self.get_market_info(market_text)
        ticker = self.extract_ticker_from_market_text(market_text)
        
        status = {
            "success": False,
            "ticker": ticker,
            "market_name": market_info["name"] if market_info else market_text,
            "error_message": None,
            "data_points": 0,
            "date_range": None,
            "attempts": 0,
            "span": span
        }
        
        try:
            self.logger.info(f"市場データ取得開始: {status['market_name']} ({ticker}) - {span}")
            
            # スパンに応じた期間調整
            adjusted_start_date, adjusted_end_date = self._adjust_date_range_for_span(start_date, end_date, span)
            
            # スパンに応じたinterval設定
            interval_map = {
                '日次': '1d',
                '週次': '1wk', 
                '月次': '1mo',
                '年次': '1y'
            }
            interval = interval_map.get(span, '1d')
            
            # リトライ機能付きデータ取得
            market_data = None
            max_retries = 3
            
            for attempt in range(max_retries):
                status["attempts"] = attempt + 1
                try:
                    self.logger.info(f"データ取得試行 {attempt + 1}/{max_retries} - interval: {interval}")
                    
                    # 複数の方法でデータ取得を試行
                    market_data = self._download_with_fallback(
                        ticker, adjusted_start_date, adjusted_end_date, interval
                    )
                    
                    if market_data is not None and not market_data.empty:
                        self.logger.info(f"取得データ形状: {market_data.shape}")
                        self.logger.info(f"取得データ列: {list(market_data.columns)}")
                        break
                    
                except Exception as e:
                    self.logger.warning(f"取得試行 {attempt + 1} 失敗: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 待機時間を増加
                    else:
                        status["error_message"] = f"データ取得失敗 (最大試行回数: {max_retries}): {str(e)}"
                        return None, status
            
            if market_data is None or market_data.empty:
                status["error_message"] = f"データが空またはNone ({ticker}, {interval})"
                return None, status
            
            # 調整後終値を取得
            close_prices = self._extract_close_prices(market_data, ticker, interval)
            if close_prices is None or close_prices.empty:
                status["error_message"] = f"調整後終値の取得に失敗 (interval: {interval})"
                return None, status
            
            # 元の期間にフィルタリング
            close_prices = self._filter_by_original_date_range(close_prices, start_date, end_date)
            if close_prices is None or close_prices.empty:
                status["error_message"] = f"指定期間でのデータが存在しません ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"
                return None, status
            
            # データ品質チェック
            clean_data = self._clean_market_data(close_prices, ticker)
            if clean_data is None or (hasattr(clean_data, 'empty') and clean_data.empty):
                status["error_message"] = "データクリーニング後にデータが空"
                return None, status
            
            # ログリターンを計算
            market_returns = self._calculate_log_returns(clean_data, ticker)
            if market_returns is None or (hasattr(market_returns, 'empty') and market_returns.empty):
                status["error_message"] = "ログリターンの計算に失敗"
                return None, status
            
            # ステータス更新
            status.update({
                "success": True,
                "data_points": len(market_returns),
                "date_range": (market_returns.index[0], market_returns.index[-1]),
                "error_message": None
            })
            
            self.logger.info(f"市場データ取得成功: {status['market_name']} - {len(market_returns)}件 ({span})")
            return market_returns, status
            
        except Exception as e:
            error_msg = f"市場データ取得中の予期しないエラー ({ticker}, {span}): {str(e)}"
            self.logger.error(error_msg)
            status["error_message"] = error_msg
            return None, status
    
    def _download_with_fallback(self, ticker: str, start_date: datetime, 
                               end_date: datetime, interval: str) -> Optional[pd.DataFrame]:
        """フォールバック機能付きダウンロード（Stock Splits完全対応版）"""
        
        # 方法1: Tickerオブジェクトを使用（最も安全）
        try:
            self.logger.info(f"方法1: Tickerオブジェクト使用 ({ticker})")
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval,
                auto_adjust=True,
                actions=False,  # Stock Splits と Dividends を無効化
                raise_errors=False
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"方法1成功: shape={data.shape}")
                return data
                
        except Exception as e:
            self.logger.warning(f"方法1失敗: {e}")
        
        # 方法2: download with group_by=None
        try:
            self.logger.info(f"方法2: group_by=None でダウンロード試行 ({ticker})")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval,
                auto_adjust=True,
                progress=False,
                repair=True,
                timeout=30,
                actions=False,
                group_by=None,  # MultiIndexを避ける
                threads=False
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"方法2成功: shape={data.shape}")
                return data
                
        except Exception as e:
            self.logger.warning(f"方法2失敗: {e}")
        
        # 方法3: 最小限の設定でダウンロード
        try:
            self.logger.info(f"方法3: 最小設定でダウンロード試行 ({ticker})")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval,
                progress=False,
                actions=False,
                threads=False
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"方法3成功: shape={data.shape}")
                return data
                
        except Exception as e:
            self.logger.warning(f"方法3失敗: {e}")
        
        # 方法4: 個別の試行（最後の手段）
        try:
            self.logger.info(f"方法4: 基本設定でダウンロード試行 ({ticker})")
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"方法4成功: shape={data.shape}")
                return data
                
        except Exception as e:
            self.logger.warning(f"方法4失敗: {e}")
        
        self.logger.error(f"全ての方法でダウンロードに失敗: {ticker}")
        return None
    
    def _filter_by_original_date_range(self, close_prices: pd.Series, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
        """元の指定期間でデータをフィルタリング（タイムゾーン対応版）"""
        try:
            if close_prices is None or close_prices.empty:
                return None
            
            # タイムゾーン処理
            if hasattr(close_prices.index, 'tz') and close_prices.index.tz is not None:
                self.logger.info(f"タイムゾーン付きIndex検出: {close_prices.index.tz}")
                # タイムゾーンを除去してローカライズ
                close_prices.index = close_prices.index.tz_convert("Asia/Tokyo").tz_localize(None)
            
            # 日付のみに正規化
            close_prices.index = close_prices.index.normalize()
            
            # start_dateとend_dateもdatetimeオブジェクトに変換（タイムゾーンなし）
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()
            
            # datetimeオブジェクトの場合はdate()で日付のみに変換
            if hasattr(start_date, 'date'):
                start_date = start_date.date()
            if hasattr(end_date, 'date'):
                end_date = end_date.date()
            
            # pandas Timestampに変換（タイムゾーンなし）
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            
            self.logger.info(f"期間フィルタリング条件: {start_ts} <= date <= {end_ts}")
            self.logger.info(f"データ範囲: {close_prices.index.min()} ~ {close_prices.index.max()}")
            
            # 日付範囲でフィルタリング
            mask = (close_prices.index >= start_ts) & (close_prices.index <= end_ts)
            filtered_data = close_prices[mask]
            
            self.logger.info(f"期間フィルタリング: {len(close_prices)} -> {len(filtered_data)} 件")
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"期間フィルタリングエラー: {e}")
            self.logger.error(f"close_prices.index type: {type(close_prices.index)}")
            self.logger.error(f"start_date type: {type(start_date)}, end_date type: {type(end_date)}")
            return None
    
    def _extract_close_prices(self, market_data: pd.DataFrame, ticker: str, interval: str) -> Optional[pd.Series]:
        """調整後終値を抽出（MultiIndex構造対応完全版）"""
        try:
            # DataFrameが空でないことを確認
            if market_data is None or market_data.empty:
                self.logger.error(f"市場データが空です ({ticker})")
                return None
            
            self.logger.info(f"市場データ構造: shape={market_data.shape}, columns={list(market_data.columns)}")
            self.logger.info(f"Index type: {type(market_data.index)}, Column type: {type(market_data.columns)}")
            
            close_prices = None
            
            # MultiIndex構造の場合の処理
            if isinstance(market_data.columns, pd.MultiIndex):
                self.logger.info("MultiIndex構造を検出")
                self.logger.info(f"MultiIndex levels: {[list(level) for level in market_data.columns.levels]}")
                
                # 実際の構造を確認：Level 0に価格データ、Level 1にティッカー
                # 例: ('Adj Close', '^N225'), ('Close', '^N225')
                
                try:
                    # タプル形式で直接アクセス
                    if ('Adj Close', ticker) in market_data.columns:
                        close_prices = market_data[('Adj Close', ticker)]
                        self.logger.info(f"('Adj Close', '{ticker}') を使用")
                    elif ('Close', ticker) in market_data.columns:
                        close_prices = market_data[('Close', ticker)]
                        self.logger.info(f"('Close', '{ticker}') を使用")
                    else:
                        # 利用可能なティッカーを探す
                        available_tickers = []
                        if len(market_data.columns.levels) > 1:
                            available_tickers = list(market_data.columns.levels[1])
                        elif len(market_data.columns.levels) > 0:
                            available_tickers = list(market_data.columns.levels[0])
                        
                        self.logger.info(f"利用可能ティッカー: {available_tickers}")
                        
                        # 最初の利用可能なティッカーを使用
                        if available_tickers:
                            fallback_ticker = available_tickers[0]
                            self.logger.info(f"指定ティッカー {ticker} が見つからないため、{fallback_ticker} を使用")
                            
                            if ('Adj Close', fallback_ticker) in market_data.columns:
                                close_prices = market_data[('Adj Close', fallback_ticker)]
                            elif ('Close', fallback_ticker) in market_data.columns:
                                close_prices = market_data[('Close', fallback_ticker)]
                    
                except Exception as e:
                    self.logger.warning(f"タプル形式アクセス失敗: {e}")
                    
                    # フォールバック: Level 0 から価格データを取得
                    try:
                        if len(market_data.columns.levels) >= 2:
                            # Level 1 にティッカーがある場合
                            if ticker in market_data.columns.levels[1]:
                                ticker_data = market_data.xs(ticker, level=1, axis=1)
                                if 'Adj Close' in ticker_data.columns:
                                    close_prices = ticker_data['Adj Close']
                                elif 'Close' in ticker_data.columns:
                                    close_prices = ticker_data['Close']
                            else:
                                # Level 0 にティッカーがある場合（従来の処理）
                                if ticker in market_data.columns.levels[0]:
                                    ticker_data = market_data[ticker]
                                    if 'Adj Close' in ticker_data.columns:
                                        close_prices = ticker_data['Adj Close']
                                    elif 'Close' in ticker_data.columns:
                                        close_prices = ticker_data['Close']
                    except Exception as e2:
                        self.logger.warning(f"xs方式アクセス失敗: {e2}")
            else:
                # 通常の構造の場合
                self.logger.info("通常のDataFrame構造を検出")
                
                # 利用可能なカラムを確認
                available_columns = list(market_data.columns)
                self.logger.info(f"利用可能カラム: {available_columns}")
                
                # 優先順位: Adj Close -> Close
                if 'Adj Close' in available_columns:
                    close_prices = market_data['Adj Close']
                    self.logger.info("Adj Close を使用")
                elif 'Close' in available_columns:
                    close_prices = market_data['Close']
                    self.logger.info("Close を使用")
                else:
                    # 部分マッチで探す（大文字小文字の違いに対応）
                    for col in available_columns:
                        if 'close' in col.lower():
                            close_prices = market_data[col]
                            self.logger.info(f"部分マッチで {col} を使用")
                            break
            
            if close_prices is None:
                self.logger.error(f"終値データが見つかりません ({ticker}): columns={list(market_data.columns)}")
                # デバッグ用: 利用可能な列を全て表示
                if isinstance(market_data.columns, pd.MultiIndex):
                    for i, level in enumerate(market_data.columns.levels):
                        self.logger.error(f"Level {i}: {list(level)}")
                    # 実際のカラム組み合わせも表示
                    self.logger.error(f"実際のカラム: {list(market_data.columns[:10])}")  # 最初の10個
                return None
            
            # DataFrameの場合はSeriesに変換
            if isinstance(close_prices, pd.DataFrame):
                if len(close_prices.columns) > 0:
                    close_prices = close_prices.iloc[:, 0]  # 最初の列を取得
                else:
                    self.logger.error(f"DataFrame に列がありません ({ticker})")
                    return None
            
            # Seriesに変換
            if not isinstance(close_prices, pd.Series):
                try:
                    close_prices = pd.Series(close_prices)
                except Exception as e:
                    self.logger.error(f"Seriesへの変換失敗 ({ticker}): {e}")
                    return None
            
            self.logger.info(f"終値データ取得成功: {len(close_prices)} 件 ({interval})")
            return close_prices
            
        except Exception as e:
            self.logger.error(f"終値抽出エラー ({ticker}, {interval}): {e}")
            return None
    
    def _clean_market_data(self, close_prices: pd.Series, ticker: str) -> Optional[pd.Series]:
        """市場データのクリーニング（タイムゾーン重複処理回避版）"""
        try:
            # 入力チェック
            if close_prices is None:
                self.logger.error(f"入力データがNoneです ({ticker})")
                return None
            
            if not isinstance(close_prices, pd.Series):
                self.logger.error(f"入力データがSeriesではありません ({ticker}): {type(close_prices)}")
                return None
            
            if close_prices.empty:
                self.logger.error(f"入力データが空です ({ticker})")
                return None
            
            original_length = len(close_prices)
            
            # NaN値のチェックと除去
            has_nan = close_prices.isnull().any()
            if has_nan:
                nan_count = close_prices.isnull().sum()
                self.logger.warning(f"市場データにNaN値が含まれています ({ticker}): {nan_count}件")
                close_prices = close_prices.dropna()
                if close_prices.empty:
                    self.logger.error(f"NaN除去後にデータが空になりました ({ticker})")
                    return None
            
            # ゼロ以下の値のチェックと除去
            has_invalid_values = (close_prices <= 0).any()
            if has_invalid_values:
                invalid_count = (close_prices <= 0).sum()
                self.logger.warning(f"市場データに非正値が含まれています ({ticker}): {invalid_count}件")
                close_prices = close_prices[close_prices > 0]
                if close_prices.empty:
                    self.logger.error(f"非正値除去後にデータが空になりました ({ticker})")
                    return None
            
            # タイムゾーン調整（_filter_by_original_date_rangeで既に処理済みの場合はスキップ）
            if hasattr(close_prices.index, 'tz') and close_prices.index.tz is not None:
                self.logger.info(f"タイムゾーン調整: {close_prices.index.tz} -> ローカライズ除去")
                close_prices.index = close_prices.index.tz_convert("Asia/Tokyo").tz_localize(None)
            
            # 日付のみに正規化（既に正規化済みの場合もある）
            if not all(t.time() == pd.Timestamp('00:00:00').time() for t in close_prices.index[:5]):
                close_prices.index = close_prices.index.normalize()
            
            # 重複した日付の処理（週次・月次データで発生する可能性）
            if close_prices.index.duplicated().any():
                duplicate_count = close_prices.index.duplicated().sum()
                self.logger.warning(f"重複した日付を検出、最後の値を使用 ({ticker}): {duplicate_count}件")
                close_prices = close_prices[~close_prices.index.duplicated(keep='last')]
            
            cleaned_length = len(close_prices)
            if cleaned_length < original_length:
                self.logger.info(f"データクリーニング: {original_length} -> {cleaned_length} 件 ({ticker})")
            
            return close_prices
            
        except Exception as e:
            self.logger.error(f"データクリーニングエラー ({ticker}): {e}")
            return None
    
    def _calculate_log_returns(self, close_prices: pd.Series, ticker: str) -> Optional[pd.Series]:
        """ログリターンを計算"""
        try:
            # 入力チェック
            if close_prices is None:
                self.logger.error(f"入力データがNoneです ({ticker})")
                return None
            
            if not isinstance(close_prices, pd.Series):
                self.logger.error(f"入力データがSeriesではありません ({ticker}): {type(close_prices)}")
                return None
            
            if len(close_prices) < 2:
                self.logger.error(f"ログリターン計算に十分なデータがありません ({ticker}): {len(close_prices)}件")
                return None
            
            # ソート（日付順に並べる）
            close_prices = close_prices.sort_index()
            
            # ログリターンを計算
            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            
            if log_returns.empty:
                self.logger.error(f"ログリターンの計算結果が空です ({ticker})")
                return None
            
            # 異常値チェック
            if log_returns.isnull().any() or np.isinf(log_returns).any():
                inf_count = np.isinf(log_returns).sum()
                nan_count = log_returns.isnull().sum()
                self.logger.warning(f"ログリターンに異常値が含まれています ({ticker}): inf={inf_count}, nan={nan_count}")
                log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
                
                if log_returns.empty:
                    self.logger.error(f"異常値除去後にログリターンが空になりました ({ticker})")
                    return None
            
            self.logger.info(f"ログリターン計算成功: {len(log_returns)} 件 ({ticker})")
            return log_returns
            
        except Exception as e:
            self.logger.error(f"ログリターン計算エラー ({ticker}): {e}")
            return None
    
    def get_market_status_text(self, status: Dict[str, Any]) -> str:
        """市場データステータスのテキストを生成"""
        if status["success"]:
            span_text = status.get("span", "")
            return f"利用可能 ({status['data_points']}件, {span_text})"
        else:
            return "利用不可"
    
    def get_market_status_description(self, status: Dict[str, Any]) -> str:
        """市場データステータスの詳細説明を生成"""
        if status["success"]:
            date_range = status.get("date_range")
            span_text = status.get("span", "")
            if date_range:
                start_date, end_date = date_range
                return (f"データ取得成功 ({span_text}): {status['data_points']}件 "
                       f"({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
            else:
                return f"データ取得成功 ({span_text}): {status['data_points']}件"
        else:
            error_msg = status.get("error_message", "不明なエラー")
            attempts = status.get("attempts", 0)
            span_text = status.get("span", "")
            return f"データ取得失敗 ({span_text}, 試行回数: {attempts}): {error_msg}"
    
    def validate_market_selection(self, market_text: str) -> Tuple[bool, str]:
        """市場選択の有効性を検証"""
        market_info = self.get_market_info(market_text)
        if market_info:
            return True, f"有効な市場: {market_info['name']}"
        else:
            return False, f"未知の市場: {market_text}"
    
    def get_available_markets(self) -> list:
        """利用可能な市場のリストを取得"""
        return list(self.market_portfolios.keys())
    
    def get_market_description(self, market_text: str) -> str:
        """市場の説明を取得"""
        market_info = self.get_market_info(market_text)
        if market_info:
            return (f"{market_info['name']} - {market_info['description']} "
                   f"({market_info['country']}, {market_info['currency']})")
        else:
            return market_text
    
    def add_custom_market(self, market_key: str, market_info: Dict[str, str]):
        """カスタム市場を追加"""
        portfolios = self.config.get_market_portfolios()
        portfolios[market_key] = market_info
        self.config.set('market.portfolios', portfolios)
        self.config.save()
        
        # 内部データも更新
        self.market_portfolios = portfolios
    
    def remove_market(self, market_key: str):
        """市場設定を削除"""
        portfolios = self.config.get_market_portfolios()
        if market_key in portfolios:
            del portfolios[market_key]
            self.config.set('market.portfolios', portfolios)
            self.config.save()
            
            # 内部データも更新
            self.market_portfolios = portfolios