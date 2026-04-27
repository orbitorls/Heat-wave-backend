import threading
import datetime
import os
from typing import Dict, List, Optional, Any
from src.core.logger import logger
from src.core.config import settings
from src.core.utils import to_jsonable
# นำเข้าฟังก์ชันเทรนจริงจาก Train_Ai.py
from Train_Ai import train as run_actual_train, EpochMetrics

class TrainingService:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "message": "Ready to start training.",
            "progress": 0,
            "config": None,
            "metrics": {
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "val_accuracy": [],
                "precision": [],
                "recall": [],
                "f1": []
            },
            "result": None,
            "error": None,
        }
        self.history: List[Dict] = []
        self.max_history = 30
        self.current_thread: Optional[threading.Thread] = None

    def get_status(self) -> Dict:
        with self.lock:
            return dict(self.state)

    def _set_state(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def _on_epoch_end_callback(self, metrics: EpochMetrics):
        """ฟังก์ชันที่จะถูกเรียกทุกครั้งที่จบ 1 Epoch ใน Train_Ai.py"""
        with self.lock:
            self.state["metrics"]["epochs"].append(metrics.epoch)
            self.state["metrics"]["train_loss"].append(float(metrics.train_loss))
            self.state["metrics"]["val_loss"].append(float(metrics.val_loss))
            self.state["metrics"]["val_accuracy"].append(float(metrics.val_event_f1)) # ใช้ F1 เป็นตัวแทนความแม่นยำของเหตุการณ์
            
            total_epochs = metrics.total_epochs
            self.state["progress"] = int((metrics.epoch / total_epochs) * 100)
            self.state["message"] = f"Epoch {metrics.epoch}/{total_epochs} completed."
            
        logger.info(f"Training Progress: Epoch {metrics.epoch}/{metrics.total_epochs} - Loss: {metrics.train_loss:.4f}")

    def start_training(self, config: Dict):
        with self.lock:
            if self.state["status"] == "running":
                raise RuntimeError("Training is already in progress.")
            
            # Reset state for new run
            self.state = {
                "status": "running",
                "started_at": datetime.datetime.now().isoformat(),
                "finished_at": None,
                "message": "Initializing real data and model...",
                "progress": 0,
                "config": config,
                "metrics": {
                    "epochs": [], "train_loss": [], "val_loss": [], 
                    "val_accuracy": [], "precision": [], "recall": [], "f1": []
                },
                "result": None,
                "error": None,
            }

        self.current_thread = threading.Thread(target=self._execute_training, args=(config,))
        self.current_thread.start()

    def _execute_training(self, config: Dict):
        try:
            # เรียกใช้ฟังก์ชันเทรนจริงจาก Train_Ai.py
            # ส่ง callback ไปเพื่อดึงค่า metrics ออกมาแบบ realtime
            result = run_actual_train(
                config=config,
                on_epoch_end=self._on_epoch_end_callback
            )

            if result:
                # Capture everything for the UI
                self._set_state(
                    status="completed",
                    finished_at=datetime.datetime.now().isoformat(),
                    message="Training finished successfully! Model saved.",
                    result=to_jsonable(result),
                    progress=100
                )
                logger.info(f"Training Job Completed. Test F1: {result.get('test_event_metrics', {}).get('f1', 0):.4f}")
            else:
                raise ValueError("Training finished but returned no results.")

        except Exception as e:
            logger.error(f"Actual training failed: {e}")
            self._set_state(
                status="failed",
                error=str(e),
                message=f"Training Error: {str(e)}",
                finished_at=datetime.datetime.now().isoformat()
            )

training_service = TrainingService()
