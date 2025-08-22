"""
MLflow Integration Module for ML Strategy Tracking.
Provides comprehensive experiment tracking and model management.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm


class MLflowTracker:
    """MLflow integration for tracking ML experiments and models."""
    
    def __init__(self,
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "ai-hedge-fund-ml",
                 model_registry_name: str = "ai-hedge-fund-models"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking URI (default: local file system)
            experiment_name: Name of the MLflow experiment
            model_registry_name: Name of the model registry
        """
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.experiment_name = experiment_name
        self.model_registry_name = model_registry_name
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self._setup_experiment()
        self._setup_model_registry()
        
        # Current run tracking
        self.current_run = None
        self.current_run_id = None
        
        logger.info(f"MLflowTracker initialized with experiment: {experiment_name}")
    
    def _setup_experiment(self):
        """Setup MLflow experiment."""
        try:
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment setup: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"MLflow experiment setup failed: {e}")
    
    def _setup_model_registry(self):
        """Setup model registry."""
        try:
            # Create model registry directory if it doesn't exist
            registry_path = f"./models/{self.model_registry_name}"
            os.makedirs(registry_path, exist_ok=True)
            
            logger.info(f"Model registry setup: {registry_path}")
            
        except Exception as e:
            logger.error(f"Model registry setup failed: {e}")
    
    def start_experiment(self, 
                        run_name: str,
                        tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow experiment run.
        
        Args:
            run_name: Name of the experiment run
            tags: Additional tags for the run
            
        Returns:
            Run ID
        """
        try:
            # Set default tags
            default_tags = {
                "project": "ai-hedge-fund",
                "phase": "ml-integration",
                "created_at": datetime.now().isoformat()
            }
            
            if tags:
                default_tags.update(tags)
            
            # Start run
            mlflow.start_run(run_name=run_name, tags=default_tags)
            self.current_run = mlflow.active_run()
            self.current_run_id = self.current_run.info.run_id
            
            logger.info(f"Started MLflow experiment: {run_name} (ID: {self.current_run_id})")
            return self.current_run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow experiment: {e}")
            return None
    
    def end_experiment(self):
        """End the current MLflow experiment run."""
        try:
            if self.current_run:
                mlflow.end_run()
                self.current_run = None
                self.current_run_id = None
                logger.info("MLflow experiment ended")
            
        except Exception as e:
            logger.error(f"Failed to end MLflow experiment: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run."""
        try:
            if self.current_run:
                mlflow.log_params(params)
                logger.info(f"Logged {len(params)} parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run."""
        try:
            if self.current_run:
                mlflow.log_metrics(metrics, step=step)
                logger.info(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_artifacts(self, 
                     local_path: str,
                     artifact_path: Optional[str] = None):
        """Log artifacts to current run."""
        try:
            if self.current_run and os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path)
                logger.info(f"Logged artifact: {local_path}")
            
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")
    
    def log_model_training(self,
                          model,
                          model_name: str,
                          model_type: str,
                          training_data_info: Dict[str, Any],
                          feature_importance: Optional[Dict[str, float]] = None):
        """
        Log model training information.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type of model ('xgboost', 'lightgbm', 'sklearn')
            training_data_info: Information about training data
            feature_importance: Feature importance dictionary
        """
        try:
            if not self.current_run:
                logger.warning("No active MLflow run for model logging")
                return
            
            # Log model based on type
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(model, model_name)
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(model, model_name)
            else:
                mlflow.sklearn.log_model(model, model_name)
            
            # Log model metadata
            model_info = {
                'model_name': model_name,
                'model_type': model_type,
                'training_timestamp': datetime.now().isoformat(),
                'training_data_samples': training_data_info.get('samples', 0),
                'training_data_features': training_data_info.get('features', 0)
            }
            
            # Save model info as artifact
            model_info_path = f"/tmp/{model_name}_info.json"
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.log_artifacts(model_info_path, "model_info")
            
            # Log feature importance if available
            if feature_importance:
                feature_importance_path = f"/tmp/{model_name}_feature_importance.json"
                with open(feature_importance_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                
                self.log_artifacts(feature_importance_path, "feature_importance")
            
            logger.info(f"Logged model training: {model_name} ({model_type})")
            
        except Exception as e:
            logger.error(f"Failed to log model training: {e}")
    
    def log_feature_engineering(self,
                               feature_config: Dict[str, Any],
                               feature_summary: Dict[str, Any],
                               feature_importance: Optional[Dict[str, float]] = None):
        """
        Log feature engineering information.
        
        Args:
            feature_config: Feature engineering configuration
            feature_summary: Feature engineering summary
            feature_importance: Feature importance dictionary
        """
        try:
            if not self.current_run:
                logger.warning("No active MLflow run for feature engineering logging")
                return
            
            # Log feature engineering info
            feature_info = {
                'feature_config': feature_config,
                'feature_summary': feature_summary,
                'timestamp': datetime.now().isoformat()
            }
            
            feature_info_path = f"/tmp/feature_engineering_info.json"
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            
            self.log_artifacts(feature_info_path, "feature_engineering")
            
            # Log feature importance if available
            if feature_importance:
                feature_importance_path = f"/tmp/feature_importance.json"
                with open(feature_importance_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                
                self.log_artifacts(feature_importance_path, "feature_importance")
            
            logger.info("Logged feature engineering information")
            
        except Exception as e:
            logger.error(f"Failed to log feature engineering: {e}")
    
    def log_prediction(self,
                      ticker: str,
                      prediction: float,
                      confidence: float,
                      features_used: List[str],
                      prediction_timestamp: Optional[datetime] = None):
        """
        Log prediction information.
        
        Args:
            ticker: Stock ticker
            prediction: Predicted value
            confidence: Prediction confidence
            features_used: List of features used for prediction
            prediction_timestamp: Timestamp of prediction
        """
        try:
            if not self.current_run:
                logger.warning("No active MLflow run for prediction logging")
                return
            
            prediction_info = {
                'ticker': ticker,
                'prediction': prediction,
                'confidence': confidence,
                'features_used': features_used,
                'prediction_timestamp': (prediction_timestamp or datetime.now()).isoformat()
            }
            
            # Log as metric for tracking over time
            self.log_metrics({
                f'prediction_{ticker}': prediction,
                f'confidence_{ticker}': confidence
            })
            
            # Log detailed info as artifact
            prediction_info_path = f"/tmp/prediction_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(prediction_info_path, 'w') as f:
                json.dump(prediction_info, f, indent=2)
            
            self.log_artifacts(prediction_info_path, "predictions")
            
            logger.info(f"Logged prediction for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def log_model_version(self,
                         model_name: str,
                         model_path: str,
                         version_description: str,
                         performance_metrics: Dict[str, float]):
        """
        Log model version to registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to saved model
            version_description: Description of this version
            performance_metrics: Performance metrics for this version
        """
        try:
            # Create model registry entry
            registry_entry = {
                'model_name': model_name,
                'version_timestamp': datetime.now().isoformat(),
                'version_description': version_description,
                'model_path': model_path,
                'performance_metrics': performance_metrics
            }
            
            # Save to registry
            registry_path = f"./models/{self.model_registry_name}/{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(registry_path, 'w') as f:
                json.dump(registry_entry, f, indent=2)
            
            logger.info(f"Logged model version: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log model version: {e}")
    
    def get_experiment_history(self,
                              experiment_name: Optional[str] = None,
                              max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get experiment history.
        
        Args:
            experiment_name: Name of experiment (default: current)
            max_results: Maximum number of results to return
            
        Returns:
            List of experiment runs
        """
        try:
            exp_name = experiment_name or self.experiment_name
            experiment = mlflow.get_experiment_by_name(exp_name)
            
            if experiment is None:
                logger.warning(f"Experiment not found: {exp_name}")
                return []
            
            # Get runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results
            )
            
            # Convert to list of dictionaries
            run_history = []
            for _, run in runs.iterrows():
                run_info = {
                    'run_id': run['run_id'],
                    'run_name': run['tags'].get('mlflow.runName', 'Unknown'),
                    'status': run['status'],
                    'start_time': run['start_time'],
                    'end_time': run['end_time'],
                    'metrics': run.filter(like='metrics.').to_dict(),
                    'params': run.filter(like='params.').to_dict()
                }
                run_history.append(run_info)
            
            logger.info(f"Retrieved {len(run_history)} experiment runs")
            return run_history
            
        except Exception as e:
            logger.error(f"Failed to get experiment history: {e}")
            return []
    
    def get_model_registry(self) -> Dict[str, Any]:
        """
        Get model registry information.
        
        Returns:
            Model registry information
        """
        try:
            registry_path = f"./models/{self.model_registry_name}"
            
            if not os.path.exists(registry_path):
                return {'models': [], 'total_models': 0}
            
            # Get all model files
            model_files = [f for f in os.listdir(registry_path) if f.endswith('.json')]
            
            models = []
            for model_file in model_files:
                try:
                    with open(os.path.join(registry_path, model_file), 'r') as f:
                        model_info = json.load(f)
                    models.append(model_info)
                except Exception as e:
                    logger.warning(f"Failed to load model info from {model_file}: {e}")
            
            # Sort by timestamp
            models.sort(key=lambda x: x.get('version_timestamp', ''), reverse=True)
            
            registry_info = {
                'models': models,
                'total_models': len(models),
                'registry_path': registry_path
            }
            
            logger.info(f"Retrieved {len(models)} models from registry")
            return registry_info
            
        except Exception as e:
            logger.error(f"Failed to get model registry: {e}")
            return {'models': [], 'total_models': 0}
    
    def get_tracker_summary(self) -> Dict[str, Any]:
        """Get comprehensive tracker summary."""
        try:
            # Get experiment info
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id if experiment else None
            
            # Get recent runs
            recent_runs = self.get_experiment_history(max_results=10)
            
            # Get model registry
            model_registry = self.get_model_registry()
            
            summary = {
                'tracking_uri': self.tracking_uri,
                'experiment_name': self.experiment_name,
                'experiment_id': experiment_id,
                'model_registry_name': self.model_registry_name,
                'current_run_id': self.current_run_id,
                'recent_runs': recent_runs,
                'model_registry': model_registry,
                'tracker_status': 'active' if self.current_run else 'idle'
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get tracker summary: {e}")
            return {'error': str(e)} 