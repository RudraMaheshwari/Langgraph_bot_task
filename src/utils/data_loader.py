import json
import logging
from typing import List
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_course_data(path: str = "src/data/courses.json") -> List[Document]:
    """Load course data from JSON file and convert to Document objects."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            course_data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Course data file not found at {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON format in {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load course data: {e}")

    if not isinstance(course_data, list):
        raise RuntimeError("Course data must be a list of course objects")

    docs = []
    skipped = 0

    for item in course_data:
        if not isinstance(item, dict):
            skipped += 1
            continue

        subjects = item.get("subjects", [])
        if isinstance(subjects, str):
            subjects = [s.strip() for s in subjects.split(",") if s.strip()]

        grades = item.get("grades", [])
        if isinstance(grades, str):
            grades = [g.strip() for g in grades.split(",") if g.strip()]

        content = (
            f"courseId: {item.get('courseId', 'N/A')}\n"
            f"Title: {item.get('title', 'N/A')}\n"
            f"Description: {item.get('description', 'N/A').strip()}\n"
            f"Subjects: {', '.join(subjects)}\n"
            f"Grade: {', '.join(grades)}\n"
            f"isDualCredit: {item.get('isDualCredit', False)}\n"
            f"isCreditRecovery: {item.get('isCreditRecovery', False)}\n"
            f"HigherEdCredits: {item.get('higherEdCredits', 0)}"
        )

        metadata = {
            "courseId": item.get("courseId", "N/A"),
            "title": item.get("title", "N/A"),
            "subjects": subjects,
            "grades": grades,
            "isDualCredit": item.get("isDualCredit", False),
            "isCreditRecovery": item.get("isCreditRecovery", False)
        }

        docs.append(Document(page_content=content, metadata=metadata))

    logger.info(f"Loaded {len(docs)} course documents")
    if skipped:
        logger.warning(f"Skipped {skipped} invalid course items")

    return docs
