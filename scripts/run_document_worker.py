from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.document_task_service import DocumentTaskService


def main() -> None:
    print("document worker started")
    DocumentTaskService.run_forever()


if __name__ == "__main__":
    main()
