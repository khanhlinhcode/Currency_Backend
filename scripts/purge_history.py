"""Delete aged prediction records and their images; run as a scheduled job."""

import argparse
from datetime import datetime, timedelta

from api.database import SessionLocal
from api.models import PredictionHistory
from api.storage import delete_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, required=True)
    args = parser.parse_args()
    if args.days < 1:
        raise SystemExit("--days must be positive")
    cutoff = datetime.utcnow() - timedelta(days=args.days)
    deleted = 0
    with SessionLocal() as db:
        while True:
            records = (
                db.query(PredictionHistory)
                .filter(PredictionHistory.created_at < cutoff)
                .order_by(PredictionHistory.id)
                .limit(100).all()
            )
            if not records:
                break
            for record in records:
                if record.image_path:
                    delete_image(record.image_path)
                db.delete(record)
                deleted += 1
            db.commit()
    print(f"Deleted {deleted} records older than {args.days} days")


if __name__ == "__main__":
    main()
