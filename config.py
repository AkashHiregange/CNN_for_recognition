from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectConfig:
    #class for consistent definitions of following parameters
    random_state: int = 22
    seed: int = 123
    test_size: float = 0.20
    cv_folds: int = 5
    output_dir: Path = Path("artifacts")
    plot_dir: Path = Path("reports/figures")

#     def create_directories(self):
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.plot_dir.mkdir(parents=True, exist_ok=True)