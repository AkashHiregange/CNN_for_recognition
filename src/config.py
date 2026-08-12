from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectConfig:
    #class for consistent definitions of following parameters
    random_state: int = 22
    seed: int = 123
    #test_size: float = 0.20
    #cv_folds: int = 5
    model_dir: Path = Path("results/models")
    plot_dir: Path = Path("results/figures")

    def create_directories(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)