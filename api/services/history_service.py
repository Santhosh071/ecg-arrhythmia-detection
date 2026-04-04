def get_patient_history(db, patient_id: str, limit: int) -> dict:
    sessions = db.get_patient_history(patient_id, limit=limit)
    return {
        "patient_id": patient_id,
        "sessions": sessions,
        "total": len(sessions),
    }


def get_patient_alerts(db, patient_id: str, limit: int) -> dict:
    alerts = db.get_recent_alerts(patient_id, limit=limit)
    return {
        "patient_id": patient_id,
        "alerts": alerts,
        "total": len(alerts),
    }


def get_patient_reports(db, patient_id: str, limit: int) -> dict:
    reports = db.get_patient_reports(patient_id, limit=limit)
    return {
        "patient_id": patient_id,
        "reports": reports,
        "total": len(reports),
    }


def get_patient_trend(db, patient_id: str, last_n_sessions: int) -> dict:
    trend = db.get_anomaly_trend(patient_id, last_n_sessions=last_n_sessions)
    return {"patient_id": patient_id, **trend}


def update_patient_alert_review(
    db,
    patient_id: str,
    alert_id: int,
    review_status: str,
    reviewer_note: str,
    reviewed_by: str | None,
) -> dict:
    alert = db.update_alert_review(
        patient_id=patient_id,
        alert_id=alert_id,
        review_status=review_status,
        reviewer_note=reviewer_note,
        reviewed_by=reviewed_by,
    )
    return {
        "patient_id": patient_id,
        "alert": alert,
    }
