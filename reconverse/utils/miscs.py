import re
from datetime import timedelta
from typing import Union

from lightning.pytorch.callbacks import RichProgressBar
from rich.progress import ProgressColumn
from rich.style import Style
from rich.text import Text

class RemainingTimeColumn(ProgressColumn):
    """Show total remaining time in training"""

    max_refresh = 1.0

    def __init__(self, style: Union[str, Style]) -> None:
        self.style = style
        self.estimated_time_per_epoch = None
        super().__init__()

    def render(self, task) -> Text:
        if 'Epoch' in task.description:
            # fetch current epoch number from task description
            m = re.search(r'Epoch (\d+)/(\d+)', task.description)
            current_epoch, total_epoch = int(m.group(1)), int(m.group(2))

            elapsed = task.finished_time if task.finished else task.elapsed
            remaining = task.time_remaining
            
            if remaining:
                time_per_epoch = elapsed + remaining
                if self.estimated_time_per_epoch is None:
                    self.estimated_time_per_epoch = time_per_epoch
                else:
                    # smooth the time_per_epoch estimation
                    self.estimated_time_per_epoch = 0.99 * self.estimated_time_per_epoch + 0.01 * time_per_epoch

                remaining_total = self.estimated_time_per_epoch * (total_epoch - current_epoch - 1) + remaining

                return Text(f"{timedelta(seconds=int(remaining_total))}", style=self.style)
            
        else:
            return Text("")


class BetterProgressBar(RichProgressBar):
    def configure_columns(self, trainer) -> list:
        columns = super().configure_columns(trainer)
        columns.insert(4, RemainingTimeColumn(style=self.theme.time))
        return columns