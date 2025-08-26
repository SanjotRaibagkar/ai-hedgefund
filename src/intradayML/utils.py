"""
Intraday ML Utils
Utility functions and helpers for intraday ML prediction system.
"""

import sys
import os
sys.path.append('./src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from .data_collector import IntradayDataCollector
from .feature_engineer import IntradayFeatureEngineer
from .model_trainer import IntradayMLTrainer
from .predictor import IntradayPredictor


class IntradayUtils:
    """Utility functions for intraday ML prediction system."""
    
    def __init__(self):
        """Initialize the utils class."""
        logger.info("🚀 Intraday Utils initialized")
    
    @staticmethod
    def validate_data_quality(data_df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        Validate data quality for a DataFrame.
        
        Args:
            data_df: DataFrame to validate
            data_type: Type of data (options, index, fii_dii, vix)
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'data_type': data_type,
                'total_rows': len(data_df),
                'total_columns': len(data_df.columns),
                'missing_values': {},
                'duplicate_rows': 0,
                'data_types': {},
                'quality_score': 0.0
            }
            
            if data_df.empty:
                validation_results['quality_score'] = 0.0
                return validation_results
            
            # Check missing values
            missing_values = data_df.isnull().sum()
            validation_results['missing_values'] = missing_values.to_dict()
            
            # Check duplicate rows
            validation_results['duplicate_rows'] = data_df.duplicated().sum()
            
            # Check data types
            validation_results['data_types'] = data_df.dtypes.to_dict()
            
            # Calculate quality score
            total_cells = len(data_df) * len(data_df.columns)
            missing_cells = data_df.isnull().sum().sum()
            duplicate_penalty = validation_results['duplicate_rows'] * 0.1
            
            quality_score = max(0.0, 1.0 - (missing_cells / total_cells) - duplicate_penalty)
            validation_results['quality_score'] = quality_score
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Error validating data quality: {e}")
            return {}
    
    @staticmethod
    def generate_data_report(db_path: str = "data/intraday_ml_data.duckdb") -> Dict[str, Any]:
        """
        Generate comprehensive data report for the intraday ML database.
        
        Args:
            db_path: Path to the database
            
        Returns:
            Dictionary with data report
        """
        try:
            import duckdb
            
            conn = duckdb.connect(db_path)
            report = {
                'report_date': datetime.now().isoformat(),
                'database_path': db_path,
                'tables': {},
                'overall_summary': {}
            }
            
            # Get list of tables
            tables_query = "SHOW TABLES"
            tables = conn.execute(tables_query).fetchdf()
            
            total_records = 0
            total_tables = len(tables)
            
            for _, table_row in tables.iterrows():
                table_name = table_row['name']
                
                # Get table info
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = conn.execute(count_query).fetchone()
                record_count = count_result[0] if count_result else 0
                
                # Get sample data
                sample_query = f"SELECT * FROM {table_name} LIMIT 5"
                sample_data = conn.execute(sample_query).fetchdf()
                
                # Get column info
                describe_query = f"DESCRIBE {table_name}"
                columns_info = conn.execute(describe_query).fetchdf()
                
                table_info = {
                    'record_count': record_count,
                    'columns': len(columns_info),
                    'sample_data': sample_data.to_dict('records') if not sample_data.empty else [],
                    'columns_info': columns_info.to_dict('records')
                }
                
                report['tables'][table_name] = table_info
                total_records += record_count
            
            # Overall summary
            report['overall_summary'] = {
                'total_tables': total_tables,
                'total_records': total_records,
                'database_size_mb': os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
            }
            
            conn.close()
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating data report: {e}")
            return {}
    
    @staticmethod
    def plot_feature_importance(feature_importance: Dict[str, float], top_n: int = 20, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary with feature importance
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        try:
            if not feature_importance:
                logger.warning("⚠️ No feature importance data provided")
                return
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            features, importance = zip(*top_features)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(features)), importance)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Error plotting feature importance: {e}")
    
    @staticmethod
    def plot_prediction_results(predictions_df: pd.DataFrame, save_path: str = None):
        """
        Plot prediction results.
        
        Args:
            predictions_df: DataFrame with predictions
            save_path: Path to save the plot
        """
        try:
            if predictions_df.empty:
                logger.warning("⚠️ No prediction data provided")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Prediction distribution
            prediction_counts = predictions_df['direction'].value_counts()
            axes[0, 0].pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Prediction Distribution')
            
            # Plot 2: Confidence distribution
            axes[0, 1].hist(predictions_df['confidence'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Confidence Distribution')
            
            # Plot 3: Predictions over time
            if 'timestamp' in predictions_df.columns:
                predictions_df['hour'] = predictions_df['timestamp'].dt.hour
                hourly_predictions = predictions_df.groupby('hour')['direction'].value_counts().unstack(fill_value=0)
                hourly_predictions.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_xlabel('Hour')
                axes[1, 0].set_ylabel('Number of Predictions')
                axes[1, 0].set_title('Predictions by Hour')
                axes[1, 0].legend()
            
            # Plot 4: Confidence vs Time
            if 'timestamp' in predictions_df.columns:
                axes[1, 1].scatter(predictions_df['timestamp'], predictions_df['confidence'], alpha=0.6)
                axes[1, 1].set_xlabel('Timestamp')
                axes[1, 1].set_ylabel('Confidence')
                axes[1, 1].set_title('Confidence vs Time')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ Prediction results plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Error plotting prediction results: {e}")
    
    @staticmethod
    def save_results_to_json(results: Dict[str, Any], file_path: str):
        """
        Save results to JSON file.
        
        Args:
            results: Dictionary with results
            file_path: Path to save the JSON file
        """
        try:
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, date):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj
            
            # Convert results
            json_results = json.loads(json.dumps(results, default=convert_datetime))
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"✅ Results saved to {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving results to JSON: {e}")
    
    @staticmethod
    def load_results_from_json(file_path: str) -> Dict[str, Any]:
        """
        Load results from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with loaded results
        """
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"✅ Results loaded from {file_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error loading results from JSON: {e}")
            return {}
    
    @staticmethod
    def create_training_pipeline(index_symbol: str, start_date: date, end_date: date, 
                                models_dir: str = "models/intraday_ml") -> Dict[str, Any]:
        """
        Create a complete training pipeline.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            start_date: Start date for training
            end_date: End date for training
            models_dir: Directory to save models
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"🚀 Starting training pipeline for {index_symbol}")
            
            # Initialize components
            data_collector = IntradayDataCollector()
            feature_engineer = IntradayFeatureEngineer()
            model_trainer = IntradayMLTrainer(models_dir)
            
            # Collect data
            logger.info("📊 Collecting data...")
            collected_data = data_collector.collect_all_data([index_symbol])
            
            # Train models
            logger.info("🏋️ Training models...")
            training_results = model_trainer.train_models(index_symbol, start_date, end_date)
            
            # Generate report
            pipeline_results = {
                'index_symbol': index_symbol,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'data_collection': collected_data,
                'training_results': training_results,
                'pipeline_status': 'completed'
            }
            
            # Cleanup
            data_collector.close()
            feature_engineer.close()
            model_trainer.close()
            
            logger.info(f"✅ Training pipeline completed for {index_symbol}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {e}")
            return {'pipeline_status': 'failed', 'error': str(e)}
    
    @staticmethod
    def create_prediction_pipeline(index_symbol: str, timestamp: datetime = None, 
                                 models_dir: str = "models/intraday_ml") -> Dict[str, Any]:
        """
        Create a complete prediction pipeline.
        
        Args:
            index_symbol: Index symbol (NIFTY or BANKNIFTY)
            timestamp: Timestamp for prediction (if None, uses current time)
            models_dir: Directory containing trained models
            
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info(f"🔮 Starting prediction pipeline for {index_symbol}")
            
            if timestamp is None:
                timestamp = datetime.now()
            
            # Initialize predictor
            predictor = IntradayPredictor(models_dir)
            
            # Load models
            if not predictor.load_models_for_index(index_symbol):
                logger.error(f"❌ Failed to load models for {index_symbol}")
                return {}
            
            # Make prediction
            prediction_result = predictor.predict_single(index_symbol, timestamp)
            
            # Get feature importance
            feature_importance = predictor.get_feature_importance_summary(index_symbol)
            
            # Get model performance
            model_performance = predictor.get_model_performance(index_symbol)
            
            # Create pipeline results
            pipeline_results = {
                'index_symbol': index_symbol,
                'timestamp': timestamp.isoformat(),
                'prediction': prediction_result,
                'feature_importance': feature_importance,
                'model_performance': model_performance,
                'pipeline_status': 'completed'
            }
            
            # Cleanup
            predictor.close()
            
            logger.info(f"✅ Prediction pipeline completed for {index_symbol}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Error in prediction pipeline: {e}")
            return {'pipeline_status': 'failed', 'error': str(e)}
    
    @staticmethod
    def validate_system_health() -> Dict[str, Any]:
        """
        Validate the health of the intraday ML system.
        
        Returns:
            Dictionary with system health status
        """
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'overall_status': 'healthy'
            }
            
            # Check database
            try:
                data_collector = IntradayDataCollector()
                health_status['components']['database'] = {
                    'status': 'healthy',
                    'message': 'Database connection successful'
                }
                data_collector.close()
            except Exception as e:
                health_status['components']['database'] = {
                    'status': 'unhealthy',
                    'message': f'Database error: {str(e)}'
                }
                health_status['overall_status'] = 'unhealthy'
            
            # Check feature engineer
            try:
                feature_engineer = IntradayFeatureEngineer()
                health_status['components']['feature_engineer'] = {
                    'status': 'healthy',
                    'message': 'Feature engineer initialized successfully'
                }
                feature_engineer.close()
            except Exception as e:
                health_status['components']['feature_engineer'] = {
                    'status': 'unhealthy',
                    'message': f'Feature engineer error: {str(e)}'
                }
                health_status['overall_status'] = 'unhealthy'
            
            # Check model trainer
            try:
                model_trainer = IntradayMLTrainer()
                health_status['components']['model_trainer'] = {
                    'status': 'healthy',
                    'message': 'Model trainer initialized successfully'
                }
                model_trainer.close()
            except Exception as e:
                health_status['components']['model_trainer'] = {
                    'status': 'unhealthy',
                    'message': f'Model trainer error: {str(e)}'
                }
                health_status['overall_status'] = 'unhealthy'
            
            # Check predictor
            try:
                predictor = IntradayPredictor()
                health_status['components']['predictor'] = {
                    'status': 'healthy',
                    'message': 'Predictor initialized successfully'
                }
                predictor.close()
            except Exception as e:
                health_status['components']['predictor'] = {
                    'status': 'unhealthy',
                    'message': f'Predictor error: {str(e)}'
                }
                health_status['overall_status'] = 'unhealthy'
            
            logger.info(f"✅ System health check completed: {health_status['overall_status']}")
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Error in system health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }
