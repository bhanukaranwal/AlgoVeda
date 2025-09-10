"""
Advanced Machine Learning Trading Strategies
Deep learning and ensemble methods for trading signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
import joblib
import logging
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy
from ..feature_engineering import FeatureEngineering
from ..risk_management import MLRiskManager
from ..portfolio import Portfolio
from ..utils.model_validation import ModelValidator
from ..utils.hyperparameter_optimization import HyperparameterOptimizer

logger = logging.getLogger(__name__)

class MLModelType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class SignalType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    RANKING = "ranking"

@dataclass
class MLSignal:
    symbol: str
    timestamp: pd.Timestamp
    signal: float  # -1 to 1 for classification, continuous for regression
    confidence: float  # 0 to 1
    probability_scores: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_prediction: Optional[float] = None
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    holding_period: Optional[int] = None  # Days

@dataclass
class ModelPerformanceMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    avg_win: float
    avg_loss: float
    information_ratio: float

class LSTMPredictor:
    """LSTM model for time series prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.sequence_length = config.get('sequence_length', 60)
        self.lstm_units = config.get('lstm_units', [50, 50])
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        self.validation_split = config.get('validation_split', 0.2)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture"""
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(keras.layers.LSTM(
            self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=input_shape,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        ))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
        
        # Dense layers
        model.add(keras.layers.Dense(25, activation='relu'))
        model.add(keras.layers.Dropout(self.dropout_rate))
        model.add(keras.layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Assuming first column is target
        
        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model"""
        # Prepare data
        X_scaled = self.scaler.fit_transform(np.column_stack([y, X]))
        X_sequences, y_sequences = self.prepare_sequences(X_scaled)
        
        # Build model
        self.model = self.build_model((X_sequences.shape[1], X_sequences.shape[2]))
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train model
        history = self.model.fit(
            X_sequences, y_sequences,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.is_fitted = True
        
        return {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss']),
            'best_epoch': np.argmin(history.history['val_loss']) + 1
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare last sequence for prediction
        X_scaled = self.scaler.transform(X)
        if len(X_scaled) >= self.sequence_length:
            X_sequence = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            prediction = self.model.predict(X_sequence, verbose=0)
            return prediction[0]
        else:
            return np.array([0.0])

class TransformerPredictor:
    """Transformer model for sequential prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.sequence_length = config.get('sequence_length', 100)
        self.d_model = config.get('d_model', 128)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 4)
        self.dff = config.get('dff', 512)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 50)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def positional_encoding(self, length: int, depth: int) -> tf.Tensor:
        """Create positional encoding"""
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate([
            np.sin(angle_rads), np.cos(angle_rads)
        ], axis=-1)
        
        return tf.cast(pos_encoding, dtype=tf.float32)

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build Transformer model"""
        # Input layer
        inputs = keras.layers.Input(shape=input_shape)
        
        # Positional encoding
        pos_encoding = self.positional_encoding(input_shape[0], self.d_model)
        x = keras.layers.Dense(self.d_model)(inputs)
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate
            )(x, x)
            
            # Add & Norm
            x = keras.layers.LayerNormalization()(x + attn_output)
            
            # Feed forward
            ffn_output = keras.layers.Dense(self.dff, activation='relu')(x)
            ffn_output = keras.layers.Dense(self.d_model)(ffn_output)
            ffn_output = keras.layers.Dropout(self.dropout_rate)(ffn_output)
            
            # Add & Norm
            x = keras.layers.LayerNormalization()(x + ffn_output)
        
        # Global average pooling
        x = keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        outputs = keras.layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Transformer model"""
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for transformer (samples, sequence_length, features)
        if len(X_scaled.shape) == 2:
            X_scaled = X_scaled.reshape(-1, self.sequence_length, X_scaled.shape[-1] // self.sequence_length)
        
        # Build model
        self.model = self.build_model((X_scaled.shape[1], X_scaled.shape[2]))
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_scaled, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.is_fitted = True
        
        return {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Transformer model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        if len(X_scaled.shape) == 2:
            X_scaled = X_scaled.reshape(-1, self.sequence_length, X_scaled.shape[-1] // self.sequence_length)
        
        return self.model.predict(X_scaled, verbose=0)

class EnsembleMLStrategy(BaseStrategy):
    """
    Advanced ensemble machine learning strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Strategy parameters
        self.signal_type = SignalType(config.get('signal_type', 'binary_classification'))
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 60])
        self.prediction_horizon = config.get('prediction_horizon', 5)  # Days ahead
        self.min_confidence = config.get('min_confidence', 0.6)
        self.rebalance_frequency = config.get('rebalance_frequency', 'daily')
        
        # Model configuration
        self.models_config = config.get('models', {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'xgboost': {'n_estimators': 100, 'max_depth': 6},
            'lightgbm': {'n_estimators': 100, 'max_depth': 6}
        })
        
        # Feature engineering
        self.feature_engineering = FeatureEngineering(config.get('feature_config', {}))
        self.feature_importance_threshold = config.get('feature_importance_threshold', 0.01)
        
        # Model components
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_weights = {}
        self.ensemble_method = config.get('ensemble_method', 'weighted_average')
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = {}
        
        # Training configuration
        self.train_test_split = config.get('train_test_split', 0.8)
        self.walk_forward_validation = config.get('walk_forward_validation', True)
        self.retrain_frequency = config.get('retrain_frequency', 30)  # Days
        self.last_training_date = None

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize ML models"""
        models = {}
        
        for model_name, model_config in self.models_config.items():
            if model_name == 'random_forest':
                models[model_name] = RandomForestClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 10),
                    random_state=42,
                    n_jobs=-1
                )
            
            elif model_name == 'xgboost':
                models[model_name] = xgb.XGBClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 6),
                    learning_rate=model_config.get('learning_rate', 0.1),
                    random_state=42
                )
            
            elif model_name == 'lightgbm':
                models[model_name] = lgb.LGBMClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 6),
                    learning_rate=model_config.get('learning_rate', 0.1),
                    random_state=42,
                    verbosity=-1
                )
            
            elif model_name == 'svm':
                models[model_name] = SVC(
                    C=model_config.get('C', 1.0),
                    kernel=model_config.get('kernel', 'rbf'),
                    probability=True,
                    random_state=42
                )
            
            elif model_name == 'neural_network':
                models[model_name] = MLPClassifier(
                    hidden_layer_sizes=model_config.get('hidden_layer_sizes', (100, 50)),
                    learning_rate_init=model_config.get('learning_rate', 0.001),
                    max_iter=model_config.get('max_iter', 500),
                    random_state=42
                )
            
            elif model_name == 'lstm':
                models[model_name] = LSTMPredictor(model_config)
            
            elif model_name == 'transformer':
                models[model_name] = TransformerPredictor(model_config)
        
        return models

    def prepare_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare features for ML models"""
        feature_data = {}
        
        for symbol, data in market_data.items():
            try:
                # Extract features using feature engineering
                features = self.feature_engineering.extract_features(symbol, data.copy())
                
                if features is None or len(features) < 50:
                    continue
                
                # Create target variable based on prediction horizon
                target = self.create_target_variable(data['close'], self.prediction_horizon)
                
                if len(target) != len(features):
                    min_len = min(len(target), len(features))
                    target = target.iloc[-min_len:]
                    features = features.iloc[-min_len:]
                
                # Combine features and target
                feature_df = pd.DataFrame(features)
                feature_df['target'] = target
                feature_df = feature_df.dropna()
                
                if len(feature_df) >= 100:  # Minimum samples required
                    feature_data[symbol] = feature_df
                    
            except Exception as e:
                logger.warning(f"Error preparing features for {symbol}: {e}")
                continue
        
        return feature_data

    def create_target_variable(self, prices: pd.Series, horizon: int) -> pd.Series:
        """Create target variable for prediction"""
        if self.signal_type == SignalType.BINARY_CLASSIFICATION:
            # Binary classification: 1 if price goes up, 0 if down
            future_returns = prices.pct_change(horizon).shift(-horizon)
            return (future_returns > 0).astype(int)
        
        elif self.signal_type == SignalType.MULTICLASS_CLASSIFICATION:
            # Multiclass: -1 (down), 0 (flat), 1 (up)
            future_returns = prices.pct_change(horizon).shift(-horizon)
            conditions = [
                future_returns < -0.02,  # Down more than 2%
                (future_returns >= -0.02) & (future_returns <= 0.02),  # Flat
                future_returns > 0.02   # Up more than 2%
            ]
            choices = [-1, 0, 1]
            return pd.Series(np.select(conditions, choices), index=prices.index)
        
        elif self.signal_type == SignalType.REGRESSION:
            # Regression: predict actual future returns
            return prices.pct_change(horizon).shift(-horizon)
        
        else:  # RANKING
            # Ranking: percentile rank of future returns
            future_returns = prices.pct_change(horizon).shift(-horizon)
            return future_returns.rolling(252).rank(pct=True)

    def train_models(self, feature_data: Dict[str, pd.DataFrame]) -> Dict[str, ModelPerformanceMetrics]:
        """Train all models on prepared data"""
        performance_metrics = {}
        
        # Combine all symbols data
        combined_data = pd.concat(feature_data.values(), ignore_index=True)
        
        # Separate features and target
        X = combined_data.drop('target', axis=1)
        y = combined_data['target']
        
        # Remove features with low importance
        X = self.select_features(X, y)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        # Split data for validation
        split_idx = int(len(X_scaled) * self.train_test_split)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Initialize and train models
        self.models = self.initialize_models()
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name} model...")
                
                if isinstance(model, (LSTMPredictor, TransformerPredictor)):
                    # Deep learning models
                    train_results = model.fit(X_train, y_train.values)
                    y_pred = model.predict(X_test)
                else:
                    # Scikit-learn models
                    model.fit(X_train, y_train)
                    
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                        y_pred = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba
                    else:
                        y_pred = model.predict(X_test)
                
                # Calculate performance metrics
                metrics = self.calculate_performance_metrics(y_test, y_pred)
                performance_metrics[model_name] = metrics
                
                logger.info(f"{model_name} - Accuracy: {metrics.accuracy:.3f}, "
                          f"Sharpe: {metrics.sharpe_ratio:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Calculate ensemble weights based on performance
        self.calculate_ensemble_weights(performance_metrics)
        
        self.last_training_date = datetime.now()
        
        return performance_metrics

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most important features"""
        try:
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            selected_features = feature_importance[
                feature_importance >= self.feature_importance_threshold
            ].index.tolist()
            
            logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
            
            return X[selected_features]
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using all features")
            return X

    def calculate_performance_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> ModelPerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Convert predictions to binary if needed
            if self.signal_type == SignalType.BINARY_CLASSIFICATION:
                y_pred_binary = (y_pred > 0.5).astype(int)
                accuracy = (y_true == y_pred_binary).mean()
            else:
                y_pred_binary = (y_pred > y_true.median()).astype(int)
                y_true_binary = (y_true > y_true.median()).astype(int)
                accuracy = (y_true_binary == y_pred_binary).mean()
            
            # Basic classification metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred_binary, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
            
            # Trading metrics
            returns = y_pred * y_true if self.signal_type == SignalType.REGRESSION else y_pred_binary * y_true
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(returns.cumsum())
            total_return = returns.sum()
            
            # Win/Loss metrics
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            
            # Information ratio
            information_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            return ModelPerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_return=total_return,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return ModelPerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, total_return=0.0,
                win_rate=0.0, avg_win=0.0, avg_loss=0.0, information_ratio=0.0
            )

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        try:
            excess_returns = returns - risk_free_rate / 252
            if excess_returns.std() == 0:
                return 0.0
            return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        except:
            return 0.0

    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            return drawdown.min()
        except:
            return 0.0

    def calculate_ensemble_weights(self, performance_metrics: Dict[str, ModelPerformanceMetrics]):
        """Calculate ensemble weights based on performance"""
        if not performance_metrics:
            return
        
        if self.ensemble_method == 'equal_weight':
            # Equal weights
            weight = 1.0 / len(performance_metrics)
            self.model_weights = {name: weight for name in performance_metrics.keys()}
        
        elif self.ensemble_method == 'performance_weighted':
            # Weight by performance (Sharpe ratio)
            sharpe_ratios = {name: max(metrics.sharpe_ratio, 0.1) 
                           for name, metrics in performance_metrics.items()}
            total_sharpe = sum(sharpe_ratios.values())
            self.model_weights = {name: sharpe / total_sharpe 
                                for name, sharpe in sharpe_ratios.items()}
        
        elif self.ensemble_method == 'accuracy_weighted':
            # Weight by accuracy
            accuracies = {name: max(metrics.accuracy, 0.1) 
                         for name, metrics in performance_metrics.items()}
            total_accuracy = sum(accuracies.values())
            self.model_weights = {name: acc / total_accuracy 
                                for name, acc in accuracies.items()}
        
        else:  # weighted_average (default)
            # Combined weighting
            scores = {}
            for name, metrics in performance_metrics.items():
                combined_score = (
                    metrics.accuracy * 0.3 +
                    max(metrics.sharpe_ratio, 0) * 0.4 +
                    metrics.information_ratio * 0.3
                )
                scores[name] = max(combined_score, 0.01)
            
            total_score = sum(scores.values())
            self.model_weights = {name: score / total_score 
                                for name, score in scores.items()}
        
        logger.info(f"Model weights: {self.model_weights}")

    def generate_predictions(self, current_features: pd.DataFrame) -> Dict[str, MLSignal]:
        """Generate predictions using ensemble of models"""
        signals = {}
        
        if not self.models or current_features.empty:
            return signals
        
        try:
            # Scale features
            X_scaled = self.scalers['main'].transform(current_features)
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if isinstance(model, (LSTMPredictor, TransformerPredictor)):
                        pred = model.predict(X_scaled)
                        predictions[model_name] = pred.flatten() if len(pred.shape) > 1 else pred
                        confidences[model_name] = np.abs(pred.flatten()) if len(pred.shape) > 1 else np.abs(pred)
                    else:
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(X_scaled)
                            if pred_proba.shape[1] == 2:
                                predictions[model_name] = pred_proba[:, 1]
                                confidences[model_name] = np.max(pred_proba, axis=1)
                            else:
                                predictions[model_name] = np.argmax(pred_proba, axis=1)
                                confidences[model_name] = np.max(pred_proba, axis=1)
                        else:
                            pred = model.predict(X_scaled)
                            predictions[model_name] = pred
                            confidences[model_name] = np.abs(pred)
                            
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    continue
            
            if not predictions:
                return signals
            
            # Ensemble predictions
            ensemble_pred = np.zeros(len(current_features))
            ensemble_conf = np.zeros(len(current_features))
            
            for model_name, pred in predictions.items():
                weight = self.model_weights.get(model_name, 1.0 / len(predictions))
                ensemble_pred += weight * pred
                ensemble_conf += weight * confidences[model_name]
            
            # Create signals for each symbol
            for i, (symbol, row) in enumerate(current_features.iterrows()):
                if ensemble_conf[i] >= self.min_confidence:
                    signal_value = ensemble_pred[i]
                    
                    # Convert to signal range [-1, 1]
                    if self.signal_type == SignalType.BINARY_CLASSIFICATION:
                        signal_value = 2 * signal_value - 1  # Convert [0,1] to [-1,1]
                    elif self.signal_type == SignalType.MULTICLASS_CLASSIFICATION:
                        signal_value = signal_value / max(abs(signal_value), 1)  # Normalize
                    
                    signals[symbol] = MLSignal(
                        symbol=symbol,
                        timestamp=pd.Timestamp.now(),
                        signal=signal_value,
                        confidence=ensemble_conf[i],
                        probability_scores={name: float(pred[i]) for name, pred in predictions.items()},
                        expected_return=signal_value * 0.02,  # Assume 2% potential return
                        holding_period=self.prediction_horizon
                    )
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {e}")
        
        return signals

    def should_retrain(self) -> bool:
        """Check if models need retraining"""
        if self.last_training_date is None:
            return True
        
        days_since_training = (datetime.now() - self.last_training_date).days
        return days_since_training >= self.retrain_frequency

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[MLSignal]:
        """Main signal generation method"""
        try:
            # Prepare features
            feature_data = self.prepare_features(market_data)
            
            if not feature_data:
                logger.warning("No valid feature data prepared")
                return []
            
            # Check if we need to retrain models
            if self.should_retrain():
                logger.info("Retraining ML models...")
                performance = self.train_models(feature_data)
                self.model_performance = performance
            
            # Get current features for prediction
            current_features = pd.DataFrame()
            for symbol, data in feature_data.items():
                if len(data) > 0:
                    latest_features = data.drop('target', axis=1).iloc[-1:]
                    latest_features.index = [symbol]
                    current_features = pd.concat([current_features, latest_features])
            
            # Generate predictions
            predictions = self.generate_predictions(current_features)
            
            # Convert to list format
            signals = list(predictions.values())
            
            logger.info(f"Generated {len(signals)} ML signals")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in ML strategy signal generation: {e}")
            return []

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics and performance metrics"""
        return {
            'model_performance': self.model_performance,
            'model_weights': self.model_weights,
            'last_training_date': self.last_training_date,
            'models_trained': len(self.models),
            'features_selected': len(self.scalers.get('main', {}).get('feature_names_in_', [])),
        }
