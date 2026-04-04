def generate_session_report(agent, patient_id: str, session_data: dict, db=None) -> dict:
    import json

    result = agent.run_tool_directly(
        "generate_report",
        json.dumps({"patient_id": patient_id, "session_data": session_data}),
    )
    success = "saved" in result.lower()
    file_path = result.replace("Report saved: ", "").strip() if success else ""

    if success and db is not None and file_path:
        try:
            db.save_report(
                patient_id=patient_id,
                file_path=file_path,
                session_id=session_data.get("session_id"),
                report_focus=session_data.get("report_focus"),
                summary=session_data.get("summary_text"),
                status="generated",
            )
        except Exception:
            pass

    return {
        "success": success,
        "file_path": file_path,
        "patient_id": patient_id,
        "message": result,
    }
