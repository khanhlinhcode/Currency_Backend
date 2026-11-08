"""Delete staging benchmark records and their private images by run ID."""

import argparse
import uuid

from sqlalchemy.orm import Session

from api.config import ENVIRONMENT
from api.database import engine
from api.models import PredictionHistory
from api.storage import delete_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true", help="Actually remove records and S3 objects")
    args = parser.parse_args()
    if ENVIRONMENT != "staging":
        raise SystemExit("Benchmark purge is restricted to APP_ENV=staging")
    run_id = str(uuid.UUID(args.run_id))
    with Session(engine) as session:
        records = session.query(PredictionHistory).filter(PredictionHistory.requested_by == f"benchmark:{run_id}").all()
        print(f"Benchmark {run_id}: {len(records)} records and up to {sum(bool(record.image_path) for record in records)} images")
        if not args.execute:
            print("Dry run only; pass --execute to delete")
            return
        for record in records:
            if record.image_path:
                delete_image(record.image_path)
            session.delete(record)
        session.commit()
        print(f"Deleted {len(records)} benchmark records")


if __name__ == "__main__":
    main()
